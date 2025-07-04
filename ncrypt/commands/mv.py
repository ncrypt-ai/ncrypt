import os
import sqlite3
from datetime import datetime

import cmd2

from ncrypt.utils import autocomplete, update_modified_at

parser = cmd2.Cmd2ArgumentParser(description="Upload an encrypted file to the virtual filesystem.")
parser.add_argument("source_path", help="Relative or absolute path to a file or directory.", completer=autocomplete)
parser.add_argument("dest_path", help="Relative or absolute path where the file or directory should be moved to.", completer=autocomplete)

@cmd2.with_argparser(parser)
def do_mv(self, args):
    """
    Move or rename a file in the virtual filesystem. Usage: mv <source path> <destination path>
    """
    source_path: str | None = args.source_path
    dest_path: str | None = args.dest_path

    if not source_path:
        self.perror("Missing source path")

        return

    if not dest_path:
        self.perror("Missing dest path")

        return

    if not source_path.startswith("/"):
        source_path: str = os.path.normpath(os.path.join(self.dir, source_path))

    else:
        source_path: str = os.path.normpath(source_path)

    if not dest_path.startswith("/"):
        dest_path: str = os.path.normpath(os.path.join(self.dir, dest_path))

    else:
        dest_path: str = os.path.normpath(dest_path)

    try:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, parent_id, is_dir FROM filesystem
            WHERE virtual_path = ?
        """, (source_path,))

        src_results = cursor.fetchone()

        if not src_results:
            self.perror(f"Source path does not exist: {source_path}")

            return

        src_id, src_parent_id, src_is_dir = src_results
        dest_name = os.path.basename(dest_path)
        dest_parent = os.path.dirname(dest_path)

        # Ensure the destination parent directory exists
        cursor.execute("""
            SELECT id FROM filesystem
            WHERE virtual_path = ? AND is_dir = 1
        """, (dest_parent,))
        dest_parent_results = cursor.fetchone()

        if not dest_parent_results:
            self.perror(f"Destination parent directory does not exist: {dest_parent}")

            return

        # Prevent overwriting an existing destination
        cursor.execute("""
            SELECT 1 FROM filesystem
            WHERE virtual_path = ?
        """, (dest_path,))

        if cursor.fetchone():
            self.perror(f"Destination already exists: {dest_path}")

            return

        new_parent_id = dest_parent_results[0]
        now = datetime.now()

        chunks: list[str] = dest_name.split(".")
        dest_name: str = chunks[0]

        if len(chunks) == 1:
            extension: None = None

        else:
            extension: str = chunks[-1]

        cursor.execute("""
            UPDATE filesystem
            SET name = ?, extension = ?, parent_id = ?, virtual_path = ?, modified_at = ?, is_dir = ?
            WHERE id = ?
        """, (dest_name, extension, new_parent_id, dest_path, now, src_is_dir, src_id))

        # Update modified_at for both old and new parent directories
        update_modified_at(self.conn, os.path.dirname(source_path), now)

        if dest_parent != os.path.dirname(source_path):
            update_modified_at(self.conn, dest_parent, now)

        self.conn.commit()

    except sqlite3.Error as e:
        self.perror(f"File system error: {e}")


def complete_mv(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx, False)
