import os
import sqlite3
from datetime import datetime

from ncrypt.utils import (
    DeletionError,
    autocomplete,
    delete_file,
    update_modified_at,
)


def do_rm(self, args):
    """
    Remove a file from the virtual filesystem. Usage: rmdir [-r] <relative or absolute directory path>
    """
    path: str | None = args.strip()

    if not path:
        self.perror("Missing file path")

        return

    if not path.startswith("/"):
        path: str = os.path.normpath(os.path.join(self.dir, path))

    else:
        path: str = os.path.normpath(path)

    try:
        cursor = self.conn.cursor()

        # Check the file exists and is not a directory
        cursor.execute("""
                SELECT id, path FROM filesystem
                WHERE virtual_path = ? AND is_dir = 0
            """, (path,))
        result = cursor.fetchone()

        if not result:
            self.perror(f"File does not exist: {path}")

            return

        file_id, file_path = result

        if self.is_remote:
            completed, message = delete_file(file_path)

            if not completed:
                raise DeletionError(message)

            # Delete the remote metadata files as well
            cursor.execute("""
                SELECT path FROM metadata
                WHERE id = ?
            """, (file_id,))
            metadata_results = cursor.fetchall()

            for metadata_path in metadata_results:
                completed, message = delete_file(metadata_path[0])

                if not completed:
                    raise DeletionError(message)

        else:
            os.remove(file_path)

        cursor.execute("DELETE FROM filesystem WHERE id = ?", (file_id,))
        cursor.execute("DELETE FROM metadata WHERE id = ?", (file_id,))

        # Recursively update modified_at on all parent directories and the root directory
        now: datetime = datetime.now()
        update_modified_at(self.conn, os.path.dirname(path), now)
        self.conn.commit()

    except sqlite3.Error as e:
        self.perror(f"File system error: {e}")


def complete_rm(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx, False)
