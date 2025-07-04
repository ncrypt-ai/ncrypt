import os
import shutil
import sqlite3
from datetime import datetime

import cmd2
import keyring
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from tqdm import tqdm

from ncrypt.utils import (
    SERVICE_NAME,
    USER_NAME,
    DownloadError,
    UploadError,
    autocomplete,
    download_file,
    get_password,
    update_modified_at,
    upload_file,
)


def call_autocomplete(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx, False)


parser = cmd2.Cmd2ArgumentParser(description="Rotate the KEK or the DEK.")
parser.add_argument("--kek", action="store_true", help="Rotate the key encryption key (KEK).")
parser.add_argument("--dek", metavar="remote_path", help="Rotate the data encryption key (DEK) for a single file.", completer=call_autocomplete)


@cmd2.with_argparser(parser)
def do_rot(self, args):
    """
    Rotate the key encryption key (DEK) for all files in the virtual filesystem, or rotate the
    data encryption key (DEK) for a specified file. Usage: rot --kek, rot --dek <remote_path>
    """
    if args.kek and args.dek:
        self.perror("Only one of --kek or --dek can be specified.")

        return

    if not args.kek and not args.dek:
        self.perror("Must specify either --kek or --dek <remote_path>.")

        return

    try:
        cursor = self.conn.cursor()
        now = datetime.now()

        if args.kek:
            old_kek = get_password()
            new_kek = os.urandom(32)
            keyring.set_password(SERVICE_NAME, USER_NAME, new_kek.hex())

            cursor.execute("""
                            SELECT id, dek FROM filesystem
                            WHERE is_dir = 0 AND dek IS NOT NULL
                        """)
            results = cursor.fetchall()

            for file_id, dek_hex in tqdm(results, desc="KEK rotation"):
                old_wrapped: bytes = bytes.fromhex(dek_hex)
                dek: bytes = unwrap_key(old_wrapped, old_kek)
                new_wrapped: str = wrap_key(dek, new_kek).hex()

                cursor.execute("""
                        UPDATE filesystem
                        SET dek = ? WHERE id = ?
                    """, (new_wrapped, file_id,))

        elif args.dek:
            remote_path: str = args.dek

            if not remote_path.startswith("/"):
                remote_path: str = os.path.normpath(os.path.join(self.dir, remote_path))

            else:
                remote_path: str = os.path.normpath(remote_path)

            cursor.execute("""
                SELECT id, path, dek FROM filesystem
                WHERE virtual_path = ? AND is_dir = 0
            """, (remote_path,))
            result = cursor.fetchone()

            # Check that the remote path exists and is a file
            if not result:
                self.perror(f"File not found or not a file: {remote_path}")

                return

            file_id, encrypted_path, dek_hex = result
            old_wrapped: bytes = bytes.fromhex(dek_hex)
            kek: bytes = get_password()

            old_dek: bytes = unwrap_key(old_wrapped, kek)
            new_dek: bytes = os.urandom(32)

            if self.is_remote:
                tmp_dir_path: str = os.path.join(self.root_dir, "tmp")
                tmp_file_path: str = os.path.join(tmp_dir_path, "tmp")
                os.mkdir(tmp_dir_path)

                completed, message, content = download_file(encrypted_path)

                if not completed:
                    raise DownloadError(message)

                with open(tmp_file_path, "wb") as file:
                    file.write(content)

                decrypted_file: bytes = decrypt_file(tmp_file_path, old_dek)
                shutil.rmtree(tmp_dir_path)

            else:
                decrypted_file: bytes = decrypt_file(encrypted_path, old_dek)

            encrypted_file, wrapped_dek = encrypt_file(decrypted_file, new_dek, kek)

            if self.is_remote:
                size: int = len(encrypted_file)
                completed, message, chunks = upload_file(encrypted_path, size, encrypted_file, [])

                # Do not allow partial uploads when rotating a key
                if not completed:
                    raise UploadError(message)

            else:
                with open(encrypted_path, "wb") as file:
                    file.write(encrypted_file)

            cursor.execute("""
                            UPDATE filesystem
                            SET dek = ?, modified_at = ?
                            WHERE id = ?
                        """, (wrapped_dek.hex(), now, file_id))

            update_modified_at(self.conn, os.path.dirname(remote_path), now)

        self.conn.commit()

    except (sqlite3.Error, OSError) as e:
        self.perror(f"Filesystem error: {e}")

    except Exception as e:
        self.perror(f"Rotation failed: {e}")


def wrap_key(dek: bytes, kek: bytes) -> bytes:
    iv = os.urandom(16)

    cipher = Cipher(algorithms.AES(kek), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(dek) + encryptor.finalize()

    return iv + ciphertext


def unwrap_key(ciphertext: bytes, kek: bytes) -> bytes:
    iv = ciphertext[:16]
    wrapped = ciphertext[16:]

    cipher = Cipher(algorithms.AES(kek), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    return decryptor.update(wrapped) + decryptor.finalize()


def encrypt_file(data: bytes, dek: bytes, kek: bytes) -> tuple[bytes, bytes]:
    # AES block size is 128 bits (16 bytes)
    iv = os.urandom(16)

    # Pad the data to be a multiple of 16 bytes
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()

    cipher = Cipher(algorithms.AES(dek), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    return iv + ciphertext, wrap_key(dek, kek)


def decrypt_file(path: str, dek: bytes) -> bytes:
    with open(path, "rb") as f:
        encrypted = f.read()

    iv = encrypted[:16]
    ciphertext = encrypted[16:]

    cipher = Cipher(algorithms.AES(dek), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()

    return unpadder.update(padded_data) + unpadder.finalize()
