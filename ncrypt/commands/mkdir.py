import os
import sqlite3
from datetime import datetime

from ncrypt.utils import update_modified_at


def do_mkdir(self, args):
    """
    Create a new virtual directory. Usage: mkdir <relative or absolute directory path>
    """
    path: str = args.strip()

    if not path:
        self.perror("Missing directory path")

        return

    if not path.startswith("/"):
        path: str = os.path.normpath(os.path.join(self.dir, path))

    else:
        path: str = os.path.normpath(path)

    parent_path: str = os.path.dirname(path)
    name: str = os.path.basename(path)

    try:
        cursor = self.conn.cursor()
        now = datetime.now()

        # Ensure parent directory exists
        cursor.execute("""
            SELECT id FROM filesystem
            WHERE virtual_path = ? AND is_dir = 1
        """, (parent_path,))

        parent_row = cursor.fetchone()

        if not parent_row:
            self.perror(f"Parent directory does not exist: {parent_path}")

            return

        parent_id = parent_row[0]

        # Check if the directory already exists under parent
        cursor.execute("""
            SELECT 1 FROM filesystem
            WHERE parent_id = ? AND name = ?
        """, (parent_id, name))

        if cursor.fetchone():
            self.perror(f"Directory '{name}' already exists in {parent_path}")

            return

        # Insert the new directory
        cursor.execute("""
            INSERT INTO filesystem (
                parent_id, name, size, is_dir, extension,
                path, virtual_path, created_at, modified_at,
                dek, status, chunks
            ) VALUES (?, ?, 0, 1, NULL, NULL, ?, ?, ?, NULL, NULL, NULL)
        """, (parent_id, name, path, now, now))

        # Recursively update modified_at on all parent directories and the root directory
        update_modified_at(self.conn, parent_path, now)
        self.conn.commit()

    except sqlite3.Error as e:
        self.perror(f"File system error: {e}")
