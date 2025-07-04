import os
import sqlite3
from datetime import datetime

import cmd2

from ncrypt.utils import (
    DeletionError,
    autocomplete,
    delete_file,
    update_modified_at,
)

parser = cmd2.Cmd2ArgumentParser(description="Remove a directory from the virtual filesystem.")
parser.add_argument("path", help="Relative or absolute path of the directory to remove.")
parser.add_argument("-r", "--recursive", action="store_true", help="Remove non-empty directories recursively")


@cmd2.with_argparser(parser)
def do_rmdir(self, args):
    """
    Remove a directory from the virtual filesystem. Supports -r for recursive removal
    of non-empty directories. Usage: rmdir [-r] <relative or absolute directory path>
    """
    path: str = args.path

    if not path:
        self.perror("Missing directory path")

        return

    if not path.startswith("/"):
        path: str = os.path.normpath(os.path.join(self.dir, path))

    else:
        path: str = os.path.normpath(path)

    try:
        cursor = self.conn.cursor()

        # Check the directory exists and is a directory
        cursor.execute("""
                SELECT id FROM filesystem
                WHERE virtual_path = ? AND is_dir = 1
            """, (path,))
        result = cursor.fetchone()

        if not result:
            self.perror(f"Directory does not exist: {path}")

            return

        dir_id = result[0]

        # If not recursive, ensure it's empty
        if not args.recursive:
            cursor.execute("""
                    SELECT 1 FROM filesystem WHERE parent_id = ?
                """, (dir_id,))

            if cursor.fetchone():
                self.perror(f"Directory not empty: {path}. Use -r to remove recursively.")

                return

        else:
            # Select all files nested under the given directory
            cursor.execute("""
                    SELECT id, path FROM filesystem
                    WHERE virtual_path LIKE ? AND is_dir = 0
                """, (f"{path}/%",))

            result = cursor.fetchall()

            failed: list[str] = []
            failed_message: str | None = None

            for file_id, virtual_path in result:
                if self.is_remote:
                    completed, message = delete_file(virtual_path)

                    if not completed:
                        failed.append(virtual_path)
                        failed_message = message

                    # Delete the remote metadata files as well
                    cursor.execute("""
                        SELECT path FROM metadata
                        WHERE id = ?
                    """, (file_id,))
                    metadata_results = cursor.fetchall()

                    for metadata_path in metadata_results:
                        completed, message = delete_file(metadata_path[0])

                        if not completed:
                            failed.append(virtual_path)
                            failed_message = message

                else:
                    os.remove(virtual_path[0])

            if failed:
                self.pinfo("Failed to delete the following files:\n")

                for virtual_path in failed:
                    self.pinfo(virtual_path)

                raise DeletionError(failed_message)

        # Delete the directory (CASCADE in table definition handles recursive case)
        cursor.execute("DELETE FROM filesystem WHERE id = ? OR virtual_path LIKE ?", (dir_id, f"{path}/%",))

        # Recursively update modified_at on all parent directories and the root directory
        now: datetime = datetime.now()
        update_modified_at(self.conn, os.path.dirname(path), now)
        self.conn.commit()

    except sqlite3.Error as e:
        self.perror(f"File system error: {e}")


def complete_rmdir(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx)
