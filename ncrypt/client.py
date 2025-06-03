import os
from enum import Enum
import pickle
from typing import Dict, List

from ncrypt.models import CRNNv1
from ncrypt.preprocessors import DefaultPreprocessor
from ncrypt.utils import BASE_DIR, OCRModel, PreProcessor
from ncrypt.utils._interfaces import TextRegion
from ncrypt.utils._types import JobResults

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
OUT_DIR = os.path.join(os.getcwd(), "out")


class PreProcessors(Enum):
    DEFAULT = DefaultPreprocessor


class Models(Enum):
    CRNN_V1 = CRNNv1


class PostProcessors(Enum):
    pass


class NcryptClient:
    def __init__(self, model: OCRModel, preprocessor: PreProcessor, postprocessor: PostProcessors, api_key: str | None = None, artifact_dir: str = ARTIFACT_DIR, out_dir: str = OUT_DIR) -> None:
        self.model: OCRModel = model
        self.preprocessor: PreProcessor = preprocessor
        self.postprocessor: PostProcessors.value = postprocessor
        self.api_key: str = api_key
        self.artifact_dir: str = artifact_dir
        self.out_dir: str = out_dir

        os.makedirs(artifact_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

    def run_job(self, pdf_path: str, save: bool = False, save_name: str | None = None) -> tuple[list[str], list[list[str]]] | None:
        if save and not save_name:
            raise ValueError("The save_path parameter must be provided in order to save the output of the job.")

        if not os.path.exists(pdf_path):
            raise ValueError("The PDF path that was provided cannot be found.")

        pdf = self.preprocessor.load_pdf(pdf_path)
        pdf, text_regions = self.preprocessor.process(pdf)
        response = self.model.submit_job(pdf, text_regions)

        status: int = response.get("status", 500)
        page_ids: list[str] = response.get("page_ids", [])
        job_ids: list[list[str]] = response.get("job_ids", [])

        if status != 200:
            raise ConnectionError("Job failed to be processed.")

        else:
            for i in range(len(page_ids)):
                pdf.add_ids(page_ids[i], job_ids[i])
                pdf.add_text_regions(text_regions[i])

            if save:
                pdf.serialize(save_name, self.out_dir)

        return page_ids, job_ids, text_regions
    
    def get_job_status(self, pdf_path: str | None = None, page_ids: List[str] = [], job_ids: List[List[str]] = [], text_regions: List[List[TextRegion]] = []):
        if not pdf_path and (not page_ids or not job_ids or not text_regions):
            raise RuntimeError("You must specify either the path to the serialized PDF file or the list of page IDs, job IDs, and text regions returned by the run_job method.")

        if pdf_path:
            with open(pdf_path, "rb") as file:
                pdf = pickle.load(file)
                all_ids: Dict[str, List[str] | List[List[str]]] = pdf.get_page_ids()
                text_regions: List[List[TextRegion]] = pdf.get_text_regions()

                page_ids: List[str] = all_ids.get("page_ids", [])
                job_ids: List[List[str]] = all_ids.get("job_ids", [])

        # TODO: Only call get_job_status for jobs that have not been returned previously
        print(job_ids)
        response: JobResults = self.model.get_job_status(page_ids, job_ids)
        status: int = response.get("status", 500)
        num_jobs: int = response.get("num_jobs", 0)
        num_jobs_completed: int = response.get("num_jobs_completed", 0)
        
        if status != 200:
            raise ConnectionError("Could not fetch job status")

        else:
            if num_jobs == num_jobs_completed:
                pass

            print(num_jobs_completed)
            print(num_jobs)
            print(status)
            print(response)