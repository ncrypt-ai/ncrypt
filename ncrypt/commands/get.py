import os
import shutil
import sqlite3

import cmd2
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from ncrypt.utils import (
    DownloadError,
    autocomplete,
    download_file,
    get_password,
)


def call_autocomplete(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx, False)


parser = cmd2.Cmd2ArgumentParser(description="Upload an encrypted file to the virtual filesystem.")
parser.add_argument("remote_path", help="Relative or absolute path of the file to decrypt", completer=call_autocomplete)
parser.add_argument("local_path", help="Relative or absolute local path for the decrypted file", completer=cmd2.Cmd.path_complete)


@cmd2.with_argparser(parser)
def do_get(self, args):
    """
    Download a file from the virtual filesystem. Usage: get <remote_path> <local_path>
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

    local_name: str = os.path.basename(local_path)
    local_parent: str = os.path.dirname(local_path)
    filename_provided: bool = False

    if not os.path.isdir(local_parent):
        self.perror(f"Local parent directory does not exist: {local_parent}")

        return

    if local_name:
        filename_provided = True

        if os.path.isfile(filename_provided):
            self.perror(f"Local file already exists: {local_path}")

            return

    try:
        cursor = self.conn.cursor()
        cursor.execute("""
                SELECT id, path, name, extension, dek FROM filesystem
                WHERE virtual_path = ? AND is_dir = 0
            """, (remote_path,))

        result = cursor.fetchone()

        # Check that the remote path exists and is a file
        if not result:
            self.perror(f"File not found or not a file: {remote_path}")

            return

        file_id, file_path, file_name, file_extension, dek = result

        if self.is_remote:
            tmp_dir_path: str = os.path.join(self.root_dir, "tmp")
            tmp_file_path: str = os.path.join(tmp_dir_path, "tmp")
            os.mkdir(tmp_dir_path)

            completed, message, content = download_file(file_path)

            if not completed:
                raise DownloadError(message)

            with open(tmp_file_path, "wb") as file:
                file.write(content)

            decrypted_file: bytes = decrypt_file(tmp_file_path, bytes.fromhex(dek))
            shutil.rmtree(tmp_dir_path)

        else:
            decrypted_file: bytes = decrypt_file(file_path, bytes.fromhex(dek))

        local_name = local_name if filename_provided else file_name
        chunks: list[str] = local_name.split(".")

        if len(chunks) == 1 and file_extension:
            local_name += f".{file_extension}"

        with open(os.path.join(local_parent, local_name), "wb") as file:
            file.write(decrypted_file)

    except (sqlite3.Error, OSError) as e:
        self.perror(f"File system error: {e}")


def decrypt_file(path: str, wrapped_key: bytes) -> bytes:
    kek = get_password()
    dek = unwrap_key(wrapped_key, kek)

    with open(path, "rb") as f:
        encrypted = f.read()

    iv = encrypted[:16]
    ciphertext = encrypted[16:]

    cipher = Cipher(algorithms.AES(dek), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()

    return unpadder.update(padded_data) + unpadder.finalize()


def unwrap_key(ciphertext: bytes, kek: bytes) -> bytes:
    iv = ciphertext[:16]
    wrapped = ciphertext[16:]

    cipher = Cipher(algorithms.AES(kek), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    return decryptor.update(wrapped) + decryptor.finalize()
