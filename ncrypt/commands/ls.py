import os
import sqlite3
from datetime import datetime

from ncrypt.utils import autocomplete


def do_ls(self, arg):
    """
    List contents of a virtual directory. Usage: ls [optional virtual path]
    """
    path: str | None = arg.strip()

    if not path:
        path: str = self.dir

    elif not path.startswith("/"):
        path: str = os.path.normpath(os.path.join(self.dir, path))

    else:
        path: str = os.path.normpath(path)

    try:
        cursor = self.conn.cursor()

        # Verify that the directory exists and is a directory
        cursor.execute("""
            SELECT id, is_dir FROM filesystem
            WHERE virtual_path = ?
        """, (path,))

        parent_row = cursor.fetchone()

        if not parent_row:
            self.perror(f"No such directory: {path}")

            return

        parent_id, is_dir = parent_row

        if not is_dir:
            self.perror(f"Not a directory: {path}")

            return

        # Fetch direct children of the directory
        cursor.execute("""
            SELECT virtual_path, extension, is_dir, size, modified_at
            FROM filesystem
            WHERE parent_id = ?
            ORDER BY is_dir DESC, name ASC
        """, (parent_id,))

        entries = cursor.fetchall()
        self.poutput(f"Contents of {path}:\n")

        for path, extension, is_dir, size, modified_at in entries:
            entry_type = "/" if is_dir else ""
            size = f"{size}" if size else "0"
            modified_time = datetime.fromisoformat(modified_at).strftime('%Y-%m-%d %H:%M')
            file_name = os.path.basename(path)

            self.poutput(f"{size:>10} {modified_time} {file_name}{entry_type}")

    except sqlite3.Error as e:
        self.perror(f"File system error: {e}")


def complete_ls(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx)
