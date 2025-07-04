import os
import sqlite3
from datetime import datetime

from .errors import FilesystemCreationError


def create_db(path: str, is_remote: bool):
    db_name: str = "ncrypt.local" if not is_remote else "ncrypt.remote"

    try:
        conn = sqlite3.connect(os.path.join(path, db_name))
        cursor = conn.cursor()
        now = datetime.now()

        cursor.execute("""
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
        """, ("filesystem",))

        exists = cursor.fetchone()

        if not exists:
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON;")

            # Create the primary table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS filesystem (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER REFERENCES filesystem(id) ON DELETE CASCADE,
                name TEXT NOT NULL, -- The name of the file that path points to (a UUID with no extension)
                size INTEGER DEFAULT 0,
                is_dir BOOLEAN NOT NULL CHECK (is_dir IN (0, 1)),
                extension TEXT,
                path TEXT, -- Where is the file actually located (either remotely or locally)
                virtual_path TEXT NOT NULL UNIQUE, -- Where is the file in the virtual filesystem
                created_at DATETIME NOT NULL,
                modified_at DATETIME NOT NULL,
                dek TEXT,
                status TEXT,
                chunks TEXT,
        
                -- Ensure unique name within the same parent
                UNIQUE(parent_id, name)
            );
            """)

            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_filesystem_name ON filesystem(name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_filesystem_virtual_path ON filesystem(virtual_path);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_filesystem_parent_id ON filesystem(parent_id);")

            conn.commit()

            # Create the root directory
            cursor.execute("""
            INSERT INTO filesystem (
                parent_id,
                name,
                size,
                is_dir,
                extension,
                path,
                virtual_path,
                created_at,
                modified_at,
                dek,
                status,
                chunks
            ) VALUES (NULL, ?, 0, 1, NULL, NULL, ?, ?, ?, NULL, NULL, NULL)
            """,
                           ("/", "/", now, now,)
                           )

            conn.commit()

            if is_remote:
                create_metadata_table(conn)
                create_key_table(conn)

    except Exception as e:
        raise FilesystemCreationError(f"Failed to create the {'local' if not is_remote else 'remote'} filesystem: {e}")


def create_metadata_table(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()

        # Create the metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER,
                name TEXT NOT NULL,
                extension TEXT NOT NULL,
                size INTEGER DEFAULT 0,
                path TEXT NOT NULL,
                type TEXT NOT NULL,
                model TEXT,
                is_search BOOLEAN NOT NULL CHECK (is_search IN (0, 1)),
                created_at DATETIME NOT NULL,
                modified_at DATETIME NOT NULL,
                PRIMARY KEY (id, type)
            );
            """)

        conn.commit()

    except Exception as e:
        raise FilesystemCreationError(f"Failed to create the metadata table: {e}")


def create_key_table(conn: sqlite3.Connection) -> None:
    try:
        cursor = conn.cursor()

        # Create the keys table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                local_path TEXT NOT NULL,
                virtual_path TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                modified_at DATETIME NOT NULL
            );
            """)

        conn.commit()

    except Exception as e:
        raise FilesystemCreationError(f"Failed to create the metadata table: {e}")


def update_modified_at(conn: sqlite3.Connection, path: str, now: datetime) -> None:
    cursor = conn.cursor()

    while path:
        cursor.execute("""
            UPDATE filesystem
            SET modified_at = ?
            WHERE virtual_path = ? AND is_dir = 1
        """, (now, path))

        if path == "/":
            break

        path = os.path.dirname(path)

    conn.commit()
