import json
import os
import shutil
import sqlite3
from base64 import b64encode
from datetime import datetime

import cmd2
import numpy as np
from concrete import fhe
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from PIL.Image import Image

from ncrypt.metadata import (
    extract_raw_image,
    extract_raw_text,
    extract_subtitles,
    get_chunks,
    get_image_embedding,
    get_keywords,
    get_text_embedding,
    get_text_summary,
    sanitize,
    text_to_bits,
    translate,
)
from ncrypt.utils import (
    EMBED_PATH,
    SEARCH_PATH,
    DownloadError,
    ProcessingError,
    UnsupportedExtensionError,
    UploadError,
    autocomplete,
    download_file,
    get_password,
    suppress_output,
    upload_file,
)


def call_autocomplete(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx, False)


parser = cmd2.Cmd2ArgumentParser(description="Extract searchable metadata for a file in the virtual filesystem.")
parser.add_argument("remote_path", help="Relative or absolute path of the file to add metadata for", completer=call_autocomplete)
parser.add_argument(
    "--type",
    required=True,
    choices=["text", "image", "audio"],
    help="Type of the input file (text, image, or audio/video)"
)
parser.add_argument("--model", help="Optional. The name of the Hugging Face model name to use to generate metadata")
parser.add_argument(
    "-x",
    action="store_true",
    help="Perform subsequent metadata processing steps on a summary of the text rather than the original. Only "
         "possible for image and audio file types"
)
parser.add_argument(
    "-c",
    action="store_true",
    help="Split text into semantically related chunks. Only possible for image and audio file types"
)
parser.add_argument(
    "-s",
    action="store_true",
    help="Prepare the text for substring search rather than the default embedding similarity search. Only possible "
         "for text and audio file types"
)


@cmd2.with_argparser(parser)
def do_meta(self, args):
    """
    Add searchable metadata to a file already uploaded to the virtual filesystem. Only
    available in remote mode. Usage: meta <remote_path> --type <text | image | audio> --model
    <huggingface model name> -[ces]
    """
    if not self.is_remote:
        self.perror("The meta command can only be used when ncrypt is in remote mode.")

        return

    remote_path: str | None = args.remote_path
    file_type: str | None = args.type
    model_name: str | None = args.model
    is_search: bool = args.s
    make_chunks: bool = args.c
    make_summary: bool = args.x

    if not remote_path:
        self.perror("Missing remote path")

        return

    if not file_type:
        self.perror("Missing file type")

        return

    if is_search and model_name == "image":
        self.perror("Cannot perform text search on an image")

        return

    if not remote_path.startswith("/"):
        remote_path: str = os.path.normpath(os.path.join(self.dir, remote_path))

    else:
        remote_path: str = os.path.normpath(remote_path)

    try:
        cursor = self.conn.cursor()
        now: datetime = datetime.now()

        cursor.execute("""
                SELECT id, path, name, extension, dek FROM filesystem
                WHERE virtual_path = ? AND is_dir = 0
            """, (remote_path,))
        result = cursor.fetchone()

        # Check that the remote path exists and is a file
        if not result:
            self.perror(f"File not found or not a file: {remote_path}")

            return

        file_id, file_path, file_name, file_extension, dek_hex = result
        old_wrapped: bytes = bytes.fromhex(dek_hex)
        kek: bytes = get_password()
        dek: bytes = unwrap_key(old_wrapped, kek)

        tmp_dir_path: str = os.path.join(self.root_dir, "tmp")
        tmp_file_path: str = os.path.join(tmp_dir_path, "tmp")
        os.mkdir(tmp_dir_path)

        completed, message, content = download_file(file_path)

        if not completed:
            raise DownloadError(message)

        decrypted_file: bytes = decrypt_file(content, dek)

        with open(tmp_file_path, "wb") as file:
            file.write(decrypted_file)

        with suppress_output():
            if is_search:
                client = fhe.Client.load(SEARCH_PATH)

            else:
                client = fhe.Client.load(EMBED_PATH)

            key_path: str = os.path.join(self.key_dir, "search" if is_search else "embed")
            client.keys.load_if_exists_generate_and_save_otherwise(key_path)
            processed: list[str] | list[list[str]] = apply_formatting(self, client, tmp_file_path, file_extension, file_type, model_name, is_search, make_chunks, make_summary)

        encrypted: list[dict[str, str | int]] = []
        metadata_path: str = f"{file_name}.json"

        if len(processed) == 0:
            self.perror(f"Generated metadata is empty: {remote_path}")

        created_at: datetime = now
        modified_at: datetime = now

        # Check if the metadata exists already
        cursor.execute("""
                SELECT created_at, path FROM metadata
                WHERE id = ? AND type = ? AND is_search = ?
            """, (file_id, file_type, int(is_search)))
        created_result = cursor.fetchone()

        if created_result:
            created_at, metadata_path = created_result

        for idx in range(len(processed)):
            val = processed[idx]

            encrypted.append({
                "idx": idx,
                "is_chunked": 1 if make_chunks else 0,
                "encrypted": val,
            })

        encrypted_json: str = json.dumps(encrypted)
        size: int = len(encrypted_json.encode("utf-8"))
        completed, message, _ = upload_file(metadata_path, size, bytes(encrypted_json, encoding="utf-8"), [])

        if not completed:
            raise UploadError(message)

        cursor.execute("""
                INSERT OR REPLACE INTO metadata (
                    id, name, extension, path, size,
                    type, model, is_search, created_at,
                    modified_at
                )
                VALUES (?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?)
            """, (
            file_id,
            file_name,
            file_extension,
            metadata_path,
            size,
            file_type,
            model_name,
            int(is_search),
            created_at,
            modified_at
        ))

        self.conn.commit()
        shutil.rmtree(tmp_dir_path)

    except (sqlite3.Error, OSError) as e:
        self.perror(f"File system error: {e}")

    except (UnsupportedExtensionError, ProcessingError) as e:
        self.perror(f"Error generating metadata: {str(e)}")


def apply_formatting(self, client: fhe.Client, local_path: str, extension: str, file_type: str, model_name: str | None, is_search: bool, make_chunks: bool, make_summary: bool) -> list[str] | list[list[str]]:
    try:
        def process_text(text: str) -> list[str] | list[list[str]]:
            translated = translate(text)
            sanitized = sanitize(translated)
            summarized = get_text_summary(sanitized) if make_summary else sanitized
            chunks = get_chunks(summarized) if make_chunks else [summarized]
            result: list[str] | list[list[str]] = []

            for chunk in chunks:
                if is_search:
                    keywords = get_keywords(chunk)
                    chunk_enc = []

                    for word in keywords:
                        bits: np.ndarray = text_to_bits(word)
                        _, enc, _ = client.encrypt(bits, bits, 0)
                        chunk_enc.append(b64encode(enc.serialize()).decode("utf-8"))

                    result.append(chunk_enc)

                else:
                    embed = get_text_embedding(chunk)
                    _, enc = client.encrypt(embed, embed)
                    result.append(b64encode(enc.serialize()).decode("utf-8"))

            return result

        if file_type == "text":
            text: str = extract_raw_text(local_path, extension)

            return process_text(text)

        elif file_type == "audio":
            text: str = extract_subtitles(local_path, extension)

            return process_text(text)

        elif file_type == "image":
            img: Image = extract_raw_image(local_path, extension)
            embed: np.ndarray = get_image_embedding(img, model_name)
            _, enc = client.encrypt(embed, embed)

            return [b64encode(enc.serialize()).decode("utf-8")]

    except (OSError, UnsupportedExtensionError) as e:
        self.perror(f"Error processing metadata: {str(e)}")


def decrypt_file(encrypted: bytes, dek: bytes) -> bytes:
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
