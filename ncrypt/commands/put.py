import json
import os
import sqlite3
import uuid
from datetime import datetime

import cmd2
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from ncrypt.utils import (
    Chunk,
    UploadError,
    autocomplete,
    get_password,
    update_modified_at,
    upload_file,
)


def call_autocomplete(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx, False)


parser = cmd2.Cmd2ArgumentParser(description="Upload an encrypted file to the virtual filesystem.")
parser.add_argument("local_path", help="Relative or absolute path to a local file to upload.", completer=cmd2.Cmd.path_complete)
parser.add_argument("remote_path", help="Relative or absolute path where the file should be placed.", completer=call_autocomplete)


@cmd2.with_argparser(parser)
def do_put(self, args):
    """
    Upload a local file to the virtual filesystem. Usage: put <local_path> <remote_path>
    """
    local_path: str | None = args.local_path
    remote_path: str | None = args.remote_path

    if not local_path:
        self.perror("Missing local path")

    if not remote_path:
        self.perror("Missing remote path")

    if not local_path.startswith("~"):
        local_path: str = os.path.normpath(os.path.join(self.ldir, local_path))

    else:
        local_path: str = os.path.normpath(os.path.join(os.path.expanduser("~"), local_path))

    if not remote_path.startswith("/"):
        remote_path: str = os.path.normpath(os.path.join(self.dir, remote_path))

    else:
        remote_path: str = os.path.normpath(remote_path)

    # Check that local_path and remote_path are both valid
    if not os.path.isfile(local_path):
        self.perror(f"Local file does not exist: {local_path}")

        return

    remote_name: str = os.path.basename(remote_path)
    remote_parent: str = os.path.dirname(remote_path)

    try:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM filesystem
            WHERE virtual_path = ? AND is_dir = 1
        """, (remote_parent,))

        parent_row = cursor.fetchone()

        if not parent_row:
            self.perror(f"Remote parent directory does not exist: {remote_parent}")

            return

        if len(remote_name) == 0:
            self.perror("Remote path must be a filename, not a directory")

            return

        parent_id = parent_row[0]

        cursor.execute("""
            SELECT chunks, status FROM filesystem
            WHERE parent_id = ? AND virtual_path = ?
        """, (parent_id, remote_path))

        results = cursor.fetchone()

        if results:
            existing_chunks, status = results
            existing_chunks: list[Chunk] = json.loads(existing_chunks)

            if not self.is_remote or (results and status == "complete"):
                self.perror("The remote path already exists")

                return

        else:
            existing_chunks: list[Chunk] = []

        now: datetime = datetime.now()
        segments: list[str] = remote_name.split(".")

        if len(segments) == 1:
            extension = None

        else:
            extension = segments[-1]

        encrypted, wrapped_key = encrypt_file(local_path)
        size = len(encrypted)
        unique_filename: str = str(uuid.uuid4())

        if self.is_remote:
            encrypted_path: str = f"{unique_filename}.txt"
            completed, message, chunks = upload_file(encrypted_path, size, encrypted, existing_chunks)

            if not completed and len(chunks) <= 1:
                raise UploadError(message)

            status: str = "complete" if completed else "incomplete"

        else:
            encrypted_path: str = os.path.join(self.root_dir, "local_files", f"{unique_filename}.txt")
            status: str = "complete"
            chunks: Chunk | None = None

            with open(encrypted_path, "wb") as file:
                file.write(encrypted)

        cursor.execute("""
                INSERT INTO filesystem (
                    parent_id, name, size, is_dir, extension,
                    path, virtual_path, created_at, modified_at,
                    dek, status, chunks
                )
                VALUES (?, ?, ?, 0, ?,
                        ?, ?, ?, ?, ?, ?, ?)
            """, (
            parent_id,
            unique_filename,
            size,
            extension,
            encrypted_path,
            remote_path,
            now,
            now,
            wrapped_key.hex(),
            status,
            json.dumps(chunks)
        ))

        update_modified_at(self.conn, remote_parent, now)
        self.conn.commit()

    except (sqlite3.Error, OSError) as e:
        self.perror(f"File system error: {e}")


def encrypt_file(path: str) -> tuple[bytes, bytes]:
    kek: bytes = get_password()
    dek: bytes = os.urandom(32)

    # AES block size is 128 bits (16 bytes)
    iv = os.urandom(16)

    with open(path, "rb") as f:
        data = f.read()

    # Pad the data to be a multiple of 16 bytes
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()

    cipher = Cipher(algorithms.AES(dek), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    return iv + ciphertext, wrap_key(dek, kek)


def wrap_key(dek: bytes, kek: bytes) -> bytes:
    iv = os.urandom(16)

    cipher = Cipher(algorithms.AES(kek), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(dek) + encryptor.finalize()

    return iv + ciphertext
