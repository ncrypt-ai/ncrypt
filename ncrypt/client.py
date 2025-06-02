import os
from enum import Enum

from ncrypt.models import CRNNv1
from ncrypt.preprocessors import DefaultPreprocessor
from ncrypt.utils import BASE_DIR, OCRModel, PreProcessor

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

            if save:
                pdf.serialize(save_name, self.out_dir)

        return page_ids, job_ids
