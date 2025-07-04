import os
import sqlite3

from ncrypt.utils import autocomplete


def do_cd(self, args):
    """
    Change virtual working directory. Usage: cd [path]
    """
    path: str | None = args.strip()

    if not path:
        path: str = "/"

    elif not path.startswith("/"):
        path: str = os.path.normpath(os.path.join(self.dir, path))

    else:
        path: str = os.path.normpath(path)

    try:
        cursor = self.conn.cursor()
        cursor.execute("""
                SELECT id FROM filesystem
                WHERE virtual_path = ? AND is_dir = 1
            """, (path,))

        if cursor.fetchone():
            self.dir = path

        else:
            self.perror(f"No such directory: {path}")

    except sqlite3.Error as e:
        self.perror(f"File system error: {e}")


def complete_cd(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx)
