import json
import os
import re
import sqlite3
from base64 import b64decode, b64encode
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cmd2
import numpy as np
from concrete import fhe
from PIL.Image import Image

from ncrypt.metadata import (
    extract_raw_image,
    get_image_embedding,
    get_text_embedding,
    text_to_bits,
)
from ncrypt.utils import (
    EMBED_PATH,
    SCALE_FACTOR,
    SEARCH_PATH,
    SIMILARITY_THRESHOLD,
    ProcessingError,
    SearchError,
    UnsupportedExtensionError,
    autocomplete,
    search_file,
    suppress_output,
)


def call_autocomplete(self, text, line, start_idx, end_idx):
    return autocomplete(self, text, line, end_idx, True)


parser = cmd2.Cmd2ArgumentParser(description="Search for files and metadata in the virtual filesystem.")
parser.add_argument("remote_path", help="Relative or absolute path of the directory to search in", completer=call_autocomplete)

parser.add_argument("--extension", help="Filter by file extension, e.g. '.txt'")
parser.add_argument("--size", help="Filter by file size in bytes (e.g. '>100', '<=5000', '=150')")
parser.add_argument(
    "--created",
    help="Filter by creation date (MMDDYYYY, e.g. '>=06012025', '<06012025', '=06012025')"
)
parser.add_argument(
    "--modified",
    help="Filter by modified date (MMDDYYYY, e.g. '>=06012025', '<06012025', '=06012025'"
)

subparsers = parser.add_subparsers(dest="subcommand", help="Search encrypted file metadata by type")

text_parser = subparsers.add_parser("text", help="Search encrypted text files")
text_parser.add_argument(
    "words",
    help="Comma-separated list of search strings. Will not be split on commas if performing an embedding search"
)
text_parser.add_argument("--model", help="Optional. The name of the Hugging Face model name to use to generate metadata")
text_parser.add_argument("-s", action="store_true", help="Use substring search instead of semantic")

audio_parser = subparsers.add_parser("audio", help="Search encrypted audio files")
audio_parser.add_argument(
    "words",
    help="Comma-separated list of search strings. Will not be split on commas if performing an embedding search"
)
audio_parser.add_argument("--model", help="Optional. The name of the Hugging Face model name to use to generate metadata")
audio_parser.add_argument("-s", action="store_true", help="Use substring search instead of semantic")

image_parser = subparsers.add_parser("image", help="Search encrypted image files")
image_parser.add_argument("local_path", help="Local filename of the image to compare against", completer=cmd2.Cmd.path_complete)
image_parser.add_argument("--model", help="Optional. The name of the Hugging Face model name to use to generate metadata")


@cmd2.with_argparser(parser)
def do_search(self, args):
    """
    Perform a (non-recursive) search for files within a directory that match the specified parameters.

    Usage:
    search <remote_path> --extension <(optional) extension_str> --size <(optional) [<|<=|=|>=|>]file_size>
    --created <(optional) [<|<=|=|>=|>]MMDDYYY> --modified <(optional) [<|<=|=|>=|>]MMDDYYY> <(optional) subcommand>

    Subcommand Usage:
        text <comma separated list of words> -[(optional) s]
        image <local_path>
        audio <comma separated list of words> -[(optional) s]
    """
    remote_path: str | None = args.remote_path
    extension: str | None = args.extension
    extension = extension if not extension else extension.strip(".")
    size: str | None = args.size
    created: str | None = args.created
    modified: str | None = args.modified

    if not remote_path:
        self.perror("Missing remote path")

        return

    if not remote_path.startswith("/"):
        remote_path: str = os.path.normpath(os.path.join(self.dir, remote_path))

    else:
        remote_path: str = os.path.normpath(remote_path)

    if args.subcommand and not self.is_remote:
        self.perror("The 'audio', 'image', and 'text' subcommands can only be used in remote mode")

    try:
        cursor = self.conn.cursor()

        # Check the directory exists and is a directory
        cursor.execute("""
                SELECT id FROM filesystem
                WHERE virtual_path = ? AND is_dir = 1
            """, (remote_path,))
        result = cursor.fetchone()

        if not result:
            self.perror(f"Directory does not exist: {remote_path}")

            return

        parent_id = result[0]

        # Fetch direct children of the directory
        cursor.execute("""
            SELECT id, name, size, virtual_path, extension, modified_at, created_at
            FROM filesystem
            WHERE parent_id = ? AND is_dir = 0
            ORDER BY name ASC
        """, (parent_id,))

        entries: list[tuple] = cursor.fetchall()

        if extension:
            entries = filter_extension(extension, entries)

        if size:
            entries = filter_size(size, entries)

        if created:
            entries = filter_date(created, False, entries)

        if modified:
            entries = filter_date(modified, True, entries)

        # Filter files to include only those with the correct metadata type if a subcommand is provided
        if args.subcommand:
            cursor.execute("""
                SELECT id, is_search, path FROM metadata
                WHERE type = ?
            """, (args.subcommand,))

            valid_ids: dict[str, str] = {}

            for row in cursor.fetchall():
                if (args.subcommand == "image" and row[1] == 0) or (row[1] == 1 and args.s):
                    valid_ids[row[0]] = row[2]

            entries = [(*entry, valid_ids[entry[0]]) for entry in entries if entry[0] in valid_ids]
            model_name: str | None = args.model

            is_search: bool = args.subcommand != "image" and args.s
            key_path: str = os.path.join(self.key_dir, "search" if is_search else "embed")

            cursor.execute("SELECT virtual_path FROM keys WHERE local_path = ?", (key_path,))
            result = cursor.fetchone()

            if args.subcommand == "image":
                entries = filter_image(key_path, result[0], args.local_path, model_name, entries)

            else:
                entries = filter_text_audio(key_path, result[0], args.words, args.s, model_name, entries)

        self.poutput(f"Contents of {remote_path} matching search parameters:\n")

        for id, name, size, virtual_path, extension, modified_at, created_at, file_path in entries:
            full_name: str = os.path.basename(virtual_path)
            size = f"{size}" if size else "0"
            modified_time = datetime.fromisoformat(modified_at).strftime('%Y-%m-%d %H:%M')
            self.poutput(f"{size:>10} {modified_time} {full_name}")

    except (sqlite3.Error, OSError) as e:
        self.perror(f"File system error: {e}")

    except (UnsupportedExtensionError, ProcessingError, SearchError) as e:
        self.perror(f"Error searching metadata: {str(e)}")


def filter_extension(extension: str, entries: list[tuple]) -> list[tuple]:
    updated_entries: list[tuple] = []

    for id, name, size, virtual_path, file_extension, modified_at, created_at in entries:
        if extension == file_extension:
            updated_entries.append((id, name, size, virtual_path, file_extension, modified_at, created_at))

    return updated_entries


def filter_size(size: str, entries: list[tuple]) -> list[tuple]:
    updated_entries: list[tuple] = []
    match = re.match(r"^(<=|>=|<|>|=)(.+)$", size.strip())

    if not match:
        raise SearchError(f"Invalid expression: {size}. Must start with one of: '<', '<=', '=', '>', '>='")

    operator, size_limit = match.groups()
    size_limit = int(size_limit)

    for id, name, file_size, virtual_path, extension, modified_at, created_at in entries:
        file_size = int(file_size)

        if (operator == "=" and file_size == size_limit) or (operator == "<" and file_size < size_limit) or (operator == "<=" and file_size <= size_limit) or (operator == ">" and file_size > size_limit) or (operator == ">=" and file_size >= size_limit):
            updated_entries.append((id, name, file_size, virtual_path, extension, modified_at, created_at))

    return updated_entries


def filter_date(date: str, is_modified: bool, entries: list[tuple]) -> list[tuple]:
    updated_entries: list[tuple] = []
    match = re.match(r"^(<=|>=|<|>|=)(.+)$", date.strip())

    if not match:
        raise SearchError(f"Invalid expression: {date}. Must start with one of: '<', '<=', '=', '>', '>='")

    operator, date_limit = match.groups()
    date_limit = datetime.strptime(date_limit, "%m%d%Y")

    for id, name, size, virtual_path, extension, modified_at, created_at in entries:
        file_date = datetime.fromisoformat(modified_at if is_modified else created_at)

        if (operator == "=" and date_limit == file_date) or (operator == "<" and file_date < date_limit) or (operator == "<=" and file_date <= date_limit) or (operator == ">" and file_date > date_limit) or (operator == ">=" and file_date >= date_limit):
            updated_entries.append((id, name, size, virtual_path, extension, modified_at, created_at))

    return updated_entries


def filter_text_audio(key_path: str, eval_key_path: str, text: str, is_search: bool, model_name: str | None, entries: list[tuple]) -> list[tuple]:
    updated_entries: list[tuple] = []

    with suppress_output():
        client = fhe.Client.load(SEARCH_PATH if is_search else EMBED_PATH)
        client.keys.load_if_exists_generate_and_save_otherwise(key_path)

    def search_wrapper(args) -> list[tuple]:
        id, name, size, virtual_path, extension, modified_at, created_at, path = args

        if is_search:
            words: list[str] = [word.strip() for word in text.split(",")]
            hashed: list[np.ndarray] = [text_to_bits(word) for word in words]
            encrypted: list[str] = []

            for word in hashed:
                enc, _, _ = client.encrypt(word, word, 0)
                encrypted.append(b64encode(enc.serialize()).decode("utf-8"))

            _, _, starting_val = client.encrypt(text_to_bits(""), text_to_bits(""), 0)
            starting_val = b64encode(starting_val.serialize()).decode("utf-8")

        else:
            embed: np.ndarray = get_text_embedding(text, model_name)
            enc, _ = client.encrypt(embed, embed)
            encrypted: list[str] = [b64encode(enc.serialize()).decode("utf-8")]
            starting_val = None

        completed, data = search_file(path, encrypted, eval_key_path, is_search, starting_val)

        if completed:
            result: list[tuple]  = []
            data = json.loads(data)

            for chunk in data["results"]:
                idx: str = str(chunk["idx"])
                enc_val: fhe.Value = fhe.Value.deserialize(b64decode(chunk["enc_val"].encode("utf-8")))

                if bool(client.decrypt(enc_val)):
                    chunk_name: str = f"{virtual_path}{'-chunk' + idx if len(data['results']) > 1 else ''}"
                    result.append((id, name, size, chunk_name, extension, modified_at, created_at, path))

            return result

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(search_wrapper, entries))

        for result in results:
            if result:
                updated_entries.extend(result)

    return updated_entries


def filter_image(key_path: str, eval_key_path: str, local_path: str, model_name: str | None, entries: list[tuple]) -> list[tuple]:
    updated_entries: list[tuple] = []
    segments: list[str] = os.path.basename(local_path).split(".")

    if len(segments) == 1:
        extension = ""

    else:
        extension = segments[-1]

    img: Image = extract_raw_image(local_path, extension)
    embed: np.ndarray = get_image_embedding(img, model_name)
    magnitude: float = np.linalg.norm(embed)

    with suppress_output():
        client = fhe.Client.load(EMBED_PATH)
        client.keys.load_if_exists_generate_and_save_otherwise(key_path)

    def search_wrapper(args) -> list[tuple]:
        id, name, size, virtual_path, extension, modified_at, created_at, path = args
        enc_val, _ = client.encrypt(embed, embed)

        completed, data = search_file(path, [b64encode(enc_val.serialize()).decode("utf-8")], eval_key_path)

        if completed:
            result: list[tuple] = []
            data = json.loads(data)

            for chunk in data["results"]:
                idx: str = str(chunk["idx"])
                enc_val: fhe.Value = fhe.Value.deserialize(b64decode(chunk["enc_val"].encode("utf-8")))
                decrypted: int = int(client.decrypt(enc_val))
                similarity: float = decrypted / (SCALE_FACTOR * magnitude)

                if similarity >= SIMILARITY_THRESHOLD:
                    chunk_name: str = f"{virtual_path}{'-chunk' + idx if len(data['results']) > 1 else ''}"
                    result.append((id, name, size, chunk_name, extension, modified_at, created_at, path))

            return result

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(search_wrapper, entries))

        for result in results:
            if result:
                updated_entries.extend(result)

    return updated_entries
