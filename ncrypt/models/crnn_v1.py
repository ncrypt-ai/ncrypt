import io
import json
import os
import shutil
from typing import TYPE_CHECKING, Dict, List
from base64 import b64encode

import cv2
import httpx
import requests
from concrete.ml.deployment import FHEModelClient
from tqdm import tqdm
import numpy as np

from ncrypt.line import Line
from ncrypt.utils import BASE_DIR, Cv2Image, OCRModel, TextRegion
from ncrypt.utils._types import JobResults

if TYPE_CHECKING:
    from ncrypt.pdf import PDFFile


class CRNNv1(OCRModel):
    def __init__(self, url: str, artifact_dir: str, api_key: str):
        self.url = url
        self.artifact_dir = os.path.join(BASE_DIR, artifact_dir)
        self.api_key = api_key

        _, fhe_dir = self.client_specs()
        self.fhe_client = FHEModelClient(path_dir=fhe_dir, key_dir=fhe_dir)
        self.fhe_client.generate_private_and_evaluation_keys()

    @property
    def model_name(self) -> str:
        return "crnn_v1"

    @property
    def base_url(self) -> str:
        return self.url
    
    @property
    def evaluation_key(self) -> str:
        artifact_dir: str = os.path.join(self.artifact_dir, self.model_name)
        key_file: str = os.path.join(artifact_dir, "keys.json")
        chunk_size: int = 250 * 1024 ** 2

        os.makedirs(artifact_dir, exist_ok=True)

        if os.path.isfile(key_file):
            with open(key_file) as file:
                data = json.load(file)

                return data.get("key_file")

        else:
            evaluation_key = self.fhe_client.get_serialized_evaluation_keys()
            buffer = io.BytesIO(initial_bytes=evaluation_key)
            buffer.seek(0)
            num_chunks: int = len(buffer.getvalue()) // chunk_size

            response = requests.get(
                f"{self.base_url}/api/v1/upload?num_chunks={num_chunks}",
                headers={
                    "Accept": "application/json",
                    "X-API-Key": self.api_key,
                },
            )

            if response.status_code == 200:
                data = response.json()
                upload_id: str = data.get("upload_id")
                key_file: str = data.get("key_file")
                urls: str = data.get("urls", [])
                chunk_etags: list[dict[str, str]] = []

                for idx in tqdm(range(len(urls)), desc="Key Upload"):
                    upload_url: str = urls[idx]
                    chunk = buffer.read(chunk_size)

                    chunk_response = httpx.put(
                        upload_url,
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/octet-stream",
                            "X-API-Key": self.api_key,
                        },
                        data=chunk,
                        timeout=3600
                    )

                    if chunk_response.status_code != 200:
                        raise Exception(f"Upload failed for chunk {idx} with error: {chunk_response.text}")

                    etag = chunk_response.headers.get("ETag")
                    chunk_etags.append({
                        "PartNumber": idx + 1,
                        "ETag": etag.strip('"')
                    })

                complete_response = httpx.post(
                    f"{self.base_url}/api/v1/complete_upload",
                    headers={
                        "Accept": "application/json",
                        "X-API-Key": self.api_key,
                    },
                    json={
                        "upload_id": upload_id,
                        "key_file": key_file,
                        "chunk_ids": chunk_etags
                    }
                )

                if complete_response.status_code == 200:
                    with open(key_file, "w") as file:
                        json.dump(data, file)

                    return data.get("key_file")
                
                raise Exception(f"Upload completion failed with error: {complete_response.text}")

            raise Exception(f"Upload failed with error: {response.text}")

    def client_specs(self) -> tuple[dict[str, str], str]:
        artifact_dir: str = os.path.join(self.artifact_dir, self.model_name, "client")
        spec_file: str = os.path.join(artifact_dir, "client.specs.json")
        sp_file: str = os.path.join(artifact_dir, "serialized_processing.json")
        version_file: str = os.path.join(artifact_dir, "versions.json")

        os.makedirs(artifact_dir, exist_ok=True)

        if os.path.isfile(spec_file) and os.path.isfile(sp_file) and os.path.isfile(version_file):
            data = {
                "client.specs": None,
                "serialized_processing": None,
                "versions": None,
            }

            with open(spec_file) as file:
                data["client.specs"] = json.load(file)

            with open(sp_file) as file:
                data["serialized_processing"] = json.load(file)

            with open(version_file) as file:
                data["versions"] = json.load(file)

            return data, os.path.join(self.artifact_dir, self.model_name)

        else:
            response = requests.get(
                f"{self.base_url}/api/v1/client?model_name={self.model_name}",
                headers={
                    "Accept": "application/json",
                    "X-API-Key": self.api_key,
                }
            )

            if response.status_code == 200:
                data = response.json()["body"]

                with open(spec_file, "w") as file:
                    json.dump(data["client.specs"], file)

                with open(sp_file, "w") as file:
                    json.dump(data["serialized_processing"], file)

                with open(version_file, "w") as file:
                    json.dump(data["versions"], file)

                shutil.make_archive(artifact_dir, "zip", artifact_dir)

                return data, os.path.join(self.artifact_dir, self.model_name)

            return {}, ""

    def model_specs(self) -> dict[str, str]:
        artifact_dir: str = os.path.join(self.artifact_dir, self.model_name)
        spec_file: str = os.path.join(artifact_dir, "model_specs.json")

        os.makedirs(artifact_dir, exist_ok=True)

        if os.path.isfile(spec_file):
            with open(spec_file) as file:
                data = json.load(file)

            return data

        else:
            response = requests.get(
                f"{self.base_url}/api/v1/model?model_name={self.model_name}",
                headers={
                    "Accept": "application/json",
                    "X-API-Key": self.api_key,
                }
            )

            if response.status_code == 200:
                data = response.json()["body"]

                with open(spec_file, "w") as file:
                    json.dump(data, file)

                return data

            return {}        

    def submit_job(self, file: "PDFFile", text_regions: list[list[TextRegion]]) -> dict[str, int | list[str] | list[list[str]]]:
        page_ids: list[str] = []
        job_ids: list[list[str]] = []
        
        for page_num in tqdm(range(file.num_pages), desc="Job Submission"):
            region: list[Line] = text_regions[page_num]
            page_jobs: list[tuple] = []

            for i in range(len(region)):
                chunks = self.__get_chunks(region[i])
                region[i].bounding_boxes = chunks # Override the original bounding boxes with ones found to be more optimal for the model

                for x, y, w, h in self.__get_chunks(region[i]):
                    img: Cv2Image = file.get_transformation(page_num, region[i].page_attr)
                    sub_image: Cv2Image = self.__crop_image(img, x, y, w, h).reshape(1, 1, 16, 100)

                    # TODO: encrypted: bytes = self.fhe_client.quantize_encrypt_serialize(sub_image)
                    # TODO: region_jobs.append(b64encode(encrypted).decode("utf-8"))

                    normalized = sub_image * (1 / np.max(sub_image))
                    img = np.reshape(normalized, (1, 1, 16, 100))

                    page_jobs.append(json.dumps(sub_image.tolist()))

            try:
                response = httpx.post(
                    f"{self.base_url}/api/v1/detect?model_name={self.model_name}",
                    headers={
                        "Accept": "application/json",
                        "X-API-Key": self.api_key,
                    },
                    json={
                        "key_file": self.evaluation_key,
                        "jobs": page_jobs
                    },
                    timeout=None
                )

                if response.status_code == 200:
                    data = response.json()

                    job_ids.append(data.get("job_ids"))
                    page_ids.append(data.get("page_id"))

            except Exception as e:  
                print(e)

                return {
                    "page_ids": page_ids,
                    "job_ids": job_ids,
                    "status": 500
                }

        return {
            "page_ids": page_ids,
            "job_ids": job_ids,
            "status": 200 if page_ids and job_ids else 500
        }

    def get_job_status(self, page_ids: List[str], job_ids: List[List[str]]) -> JobResults | None:
        status = {}
        complete = 0
        processed = 0

        for page_num in range(len(page_ids)):
            page_id: str = page_ids[page_num]
            jobs: List[str] = job_ids[page_num]

            status[page_id] = []

            try:
                response = httpx.post(
                    f"{self.base_url}/api/v1/status",
                    headers={
                        "Accept": "application/json",
                        "X-API-Key": self.api_key,
                    },
                    json={
                        "jobs": jobs
                    },
                    timeout=None
                )

                if response.status_code == 200:
                    data = response.json()
                    page_status = data.get("status")
                    vals = []

                    for obj in page_status:
                        completed = obj.get("status", "incopmplete")
                        processed += 1
                        complete += 1 if completed == "complete" else 0

                        vals.append({
                            "status": obj.get("status"),
                            "job_id": obj.get("job_id"),
                            "output": obj.get("val", {}).get("response")
                        })

                    status[page_id] = vals

            except Exception as e:  
                print(e)

                return {
                    "status": 500
                }


        return {
            "num_jobs": processed,
            "jobs_completed": complete,
            "status": 200,
            "pages": status
        }

    @staticmethod
    def __crop_image(img: Cv2Image, x: float, y: float, w: float, h: float) -> Cv2Image:
        cropped = img.copy()[y: y + h, x: x + w]
        new_w: int = 100
        new_h: int = 16

        return cv2.resize(cropped, (new_w, new_h))

    @staticmethod
    def __get_chunks(line: Line) -> list[tuple[float, float, float, float]]:
        chunks = []
        curr_box = None
        target_aspect_ratio: float = 100 / 16

        for i in range(len(line.bounding_boxes)):
            x1, y1, w1, h1 = line.bounding_boxes[i]

            if not curr_box:
                curr_box = (x1, y1, w1, h1)

            else:
                x2, y2, w2, h2 = curr_box
                curr_aspect_ratio = w2 / h2 if h2 != 0 else float('inf')

                merged_x1 = min(x1, x2)
                merged_y1 = min(y1, y2)
                merged_x2 = max(x1 + w1, x2 + w2)
                merged_y2 = max(y1 + h1, y2 + h2)
                merged_w = merged_x2 - merged_x1
                merged_h = merged_y2 - merged_y1
                merged_aspect_ratio = merged_w / merged_h if merged_h != 0 else float('inf')

                if abs(target_aspect_ratio - curr_aspect_ratio) < abs(target_aspect_ratio - merged_aspect_ratio):
                    chunks.append(curr_box)
                    curr_box = (x1, y1, w1, h1)

                else:
                    curr_box = (merged_x1, merged_y1, merged_w, merged_h)

            if i == len(line.bounding_boxes) - 1 and curr_box:
                chunks.append(curr_box)

        self.bo

        return chunks
