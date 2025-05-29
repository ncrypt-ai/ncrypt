from typing import Dict, List, Tuple, Union, TYPE_CHECKING
import os
import json
import requests
import shutil

import cv2
from concrete.ml.deployment import FHEModelClient

from ncrypt.line import Line
from ncrypt.utils import OCRModel, TextRegion, Cv2Image, BASE_DIR

if TYPE_CHECKING:
    from ncrypt.pdf import PDFFile


class CRNNv1(OCRModel):
    def __init__(self, url: str, artifact_dir: str, api_key: str):
        self.url = url
        self.artifact_dir = os.path.join(BASE_DIR, artifact_dir)
        self.api_key = api_key

        _, fhe_dir = self.client_specs()

        self.fhe_client = FHEModelClient(path_dir=fhe_dir, key_dir=fhe_dir)
        self.evaluation_keys = self.fhe_client.get_serialized_evaluation_keys()

    @property
    def model_name(self) -> str:
        return "crnn_v1"

    @property
    def base_url(self) -> str:
        return self.url

    def client_specs(self) -> Tuple[Dict[str, str], str]:
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

            with open(spec_file, "r") as file:
                data["client.specs"] = json.load(file)

            with open(sp_file, "r") as file:
                data["serialized_processing"] = json.load(file)

            with open(version_file, "r") as file:
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

    def model_specs(self) -> Dict[str, str]:
        artifact_dir: str = os.path.join(self.artifact_dir, self.model_name)
        spec_file: str = os.path.join(artifact_dir, "model_specs.json")

        os.makedirs(artifact_dir, exist_ok=True)

        if os.path.isfile(spec_file):
            with open(spec_file, "r") as file:
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

    def submit_job(self, file: "PDFFile", text_regions: List[List[TextRegion]], api_key: str) -> Dict[str, Union[int, List[str], List[List[str]]]]:
        for page_num in range(file.num_pages):
            img: Cv2Image = file.get_page(page_num)
            region: List[Line] = file.text_regions[page_num]
            page_jobs: List[List[bytes]] = []

            for i in range(len(region)):
                region_jobs: List[bytes] = []

                for x, y, w, h in self.__get_chunks(region[i]):
                    print(x, y, w, h)
                    sub_image: Cv2Image = self.__crop_image(img, x, y, w, h)
                    encrypted: bytes = self.fhe_client.quantize_encrypt_serialize(sub_image)

                    region_jobs.append(encrypted)

                page_jobs.append(region_jobs)

            print(page_jobs)

    def get_job_status(self):
        pass

    @staticmethod
    def __crop_image(img: Cv2Image, x: float, y: float, w: float, h: float) -> Cv2Image:
        cropped = img.copy()[y: y + h, x: x + w]
        new_w: int = 100
        new_h: int = 16

        return cv2.resize(cropped, (new_w, new_h))

    @staticmethod
    def __get_chunks(line: Line) -> List[Tuple[float, float, float, float]]:
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

        return chunks
