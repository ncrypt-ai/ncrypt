import io
from typing import Literal, TypedDict

import httpx
from tqdm import tqdm

from .errors import UploadError


class Chunk(TypedDict):
    status: Literal["complete", "incomplete"] | None
    upload_id: str | None
    url: str
    chunk_size: int | None
    PartNumber: int
    ETag: str | None


def _get_file_id(path: str) -> str:
    segments: list[str] = path.split("/")

    return segments[-1]


def _filter_chunk_fields(chunks: list[Chunk]) -> list[Chunk]:
    return [Chunk(ETag=chunk["ETag"], PartNumber=chunk["PartNumber"]) for chunk in chunks]


def upload_file(path: str, num_bytes: int, content: bytes, chunks: list[Chunk]) -> tuple[bool, str, list[dict[str, str]]]:
    idx: str = _get_file_id(path)
    request_url: str = f"https://ncryptai.com/api/v1/files/upload/request?id={idx}&file_size={num_bytes}"
    complete_url: str = f"https://ncryptai.com/api/v1/files/upload/complete?id={idx}"

    # Try to resume the upload first if there are pre-existing chunks
    if len(chunks) > 0:
        return resume_upload(path, num_bytes, content, chunks)

    try:
        req_response = httpx.get(
            request_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                # "X-API-Key": API_KEY,
            },
        )

        req_data: dict[str, str] = req_response.json()

        if req_response.status_code != 200:
            return False, req_data.get("message", ""), []

        is_multipart: bool = req_data.get("upload_type", "single") == "multipart"

        if is_multipart:
            upload_urls: list[str] = req_data.get("urls", [])
            chunk_size: int = int(req_data.get("chunk_size", "0"))
            chunks: list[Chunk] = []
            success: bool = True

            buffer = io.BytesIO(initial_bytes=content)
            buffer.seek(0)

            for i in tqdm(range(len(upload_urls)), desc="Multipart File Upload"):
                upload_url: str = upload_urls[i]
                chunk = buffer.read(chunk_size)

                chunk_response = httpx.put(
                    upload_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/octet-stream",
                        # "X-API-Key": API_KEY,
                    },
                    content=chunk,
                    timeout=10000
                )
                success: bool = success and chunk_response.status_code == 200

                chunks.append({
                    "PartNumber": i + 1,
                    "url": upload_url,
                    "status": "complete" if chunk_response.status_code == 200 else "incomplete",
                    "upload_id": req_data.get("upload_id", ""),
                    "chunk_size": chunk_size,
                    "ETag": chunk_response.headers.get("ETag").strip('"')
                })

            if success:
                complete_response = httpx.post(
                    complete_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        # "X-API-Key": API_KEY,
                    },
                    json={
                        "chunk_ids": _filter_chunk_fields(chunks),
                        "upload_id": req_data.get("upload_id", "")
                    }
                )
                complete_data = complete_response.text

                return complete_response.status_code == 200, complete_data, chunks

            return False, "Failed to complete upload", chunks

        else:
            upload_url: str = req_data.get("url")
            upload_response = httpx.put(
                upload_url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/octet-stream",
                    # "X-API-Key": API_KEY,
                },
                content=content
            )

            if upload_response.status_code == 200:
                complete_response = httpx.post(
                    complete_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        # "X-API-Key": API_KEY,
                    },
                    json={
                        "chunk_ids": [],
                        "upload_id": "id"
                    }
                )
                complete_data = complete_response.text

                return complete_response.status_code == 200, complete_data, [{"PartNumber": 1, "status": "incomplete" if complete_response.status_code != 200 else "complete", "url": upload_url}]

            else:
                return False, "Failed to complete upload", [{"PartNumber": 1, "status": "incomplete", "url": upload_url}]

    except Exception as e:
        return False, f"Internal error occcured while uploading file: {str(e)}", []


def resume_upload(path: str, num_bytes: int, content: bytes, chunks: list[Chunk]) -> tuple[bool, str, list[dict[str, str]]]:
    idx: str = _get_file_id(path)
    complete_url: str = f"https://ncryptai.com/api/v1/files/upload/complete?id={idx}"
    success: bool = True

    try:
        updated_chunks: list[Chunk] = []
        buffer = io.BytesIO(initial_bytes=content)
        buffer.seek(0)

        for i in tqdm(range(len(chunks)), desc="Multipart File Upload"):
            status: str = chunks[i]["status"]

            if status == "complete":
                updated_chunks.append(chunks[i])

            else:
                part_number: int = chunks[i]["PartNumber"]
                upload_url: str = chunks[i]["url"]
                upload_id: str = chunks[i]["upload_id"]
                etag: str = chunks[i]["ETag"]
                chunk_size: int = chunks[i]["chunk_size"]

                chunk = buffer.read(chunk_size)
                chunk_response = httpx.put(
                    upload_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/octet-stream",
                        # "X-API-Key": API_KEY,
                    },
                    content=chunk,
                )
                success: bool = success and chunk_response.status_code == 200

                if success:
                    chunks.append({
                        "PartNumber": part_number,
                        "url": upload_url,
                        "status": "complete",
                        "upload_id": upload_id,
                        "chunk_size": chunk_size,
                        "ETag": etag
                    })

                else:
                    raise UploadError("Failed to complete upload")

            if i == len(chunks) - 1:
                complete_response = httpx.post(
                    complete_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        # "X-API-Key": API_KEY,
                    },
                    json={
                        "chunk_ids": _filter_chunk_fields(updated_chunks),
                        "upload_id": upload_id
                    }
                )
                complete_data = complete_response.text

                return complete_response.status_code == 200, complete_data, updated_chunks

        raise UploadError("No partial upload to resume")

    except Exception as _:
        return upload_file(path, num_bytes, content, [])


def download_file(path: str) -> tuple[bool, str, bytes]:
    idx: str = _get_file_id(path)
    download_url: str = f"https://ncryptai.com/api/v1/files/download?id={idx}"

    try:
        response = httpx.get(
            download_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                # "X-API-Key": API_KEY,
            },
        )

        data: dict[str, str] = response.json()

        if response.status_code != 200:
            return False, data.get("message", ""), b""

        url: str = data.get("url", "")
        file_response = httpx.get(url)

        return file_response.status_code == 200, "File downloaded successfully", file_response.content

    except Exception as e:
        return False, f"Internal error occurred while downloading file: {str(e)}", b""


def delete_file(path: str) -> tuple[bool, str]:
    idx: str = _get_file_id(path)
    delete_url: str = f"https://ncryptai.com/api/v1/files/delete?id={idx}"

    try:
        response = httpx.delete(
            delete_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                # "X-API-Key": API_KEY,
            },
        )

        data = response.text

        return response.status_code == 200, data

    except Exception as e:
        return False, f"Internal error occcured while deleting file: {str(e)}"


def file_exists(path: str) -> bool:
    idx: str = _get_file_id(path)
    exists_url: str = f"https://ncryptai.com/api/v1/files/exists?id={idx}"

    try:
        response = httpx.get(
            exists_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                # "X-API-Key": API_KEY,
            },
        )

        data = response.json()

        return data["exists"]

    except Exception as _:
        return False


def search_file(filename: str, enc_val: list[str], eval_key: str, is_search: bool = False, starting_val: str | None = None):
    search_url: str = "https://ncryptai.com/api/v1/files/search"

    try:
        response = httpx.post(
            search_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                # "X-API-Key": API_KEY,
            },
            json={
                "filename": filename,
                "is_search": is_search,
                "enc_val": enc_val,
                "starting_val": starting_val,
                "eval_key": eval_key
            },
            timeout=None
        )

        data = response.text

        return response.status_code == 200, data

    except Exception as e:
        return False, f"Internal error occcured while searching file: {str(e)}"
