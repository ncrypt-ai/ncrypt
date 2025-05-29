import os
from enum import Enum
from typing import Union, Tuple, List

from ncrypt.utils import BASE_DIR, OCRModel, PreProcessor
from ncrypt.preprocessors import DefaultPreprocessor
from ncrypt.models import CRNNv1


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

    def run_job(self, pdf_path: str, save: bool = False, save_path: Union[str, None] = None) -> Union[Tuple[List[str], List[List[str]]], None]:
        if save and not save_path:
            raise ValueError("The save_path parameter must be provided in order to save the output of the job.")

        if not os.path.exists(pdf_path):
            raise ValueError("The PDF path that was provided cannot be found.")

        pdf = self.preprocessor.load_pdf(pdf_path)
        pdf, text_regions = self.preprocessor.process(pdf)
        response = self.model.submit_job(pdf, text_regions, self.api_key)

        status: int = response.get("status", 500)
        page_ids: List[str] = response.get("page_ids", [])
        job_ids: List[List[str]] = response.get("job_ids", [])

        if status != 200:
            raise ConnectionError("Job failed to be processed.")

        else:
            for i in range(len(page_ids)):
                pdf.add_ids(page_ids[i], job_ids[i])

            if save:
                os.makedirs(save_path, exist_ok=True)
                pdf.serialize(save_path)

        return page_ids, job_ids



    # def get_job_status(self, file: PDFFile, text_regions: List[List[TextRegion]], api_key: str) -> Tuple[PDFFile, Dict[str, str]]:
    #     pass

# Model has the iterator for a given text region and the get_img function
# Client is responsible for updating the PDF after preprocessing and post-processing is done
# Client is responsible for retries

# Allow setting the vocabulary in advance, or allow a custom language model
# Assign a preprocessor and a postprocessor
# Assign a model. Once assigned, it checks to see whether or not you have its specs saved already otherwise it fetches them

# TODO: Finish client
# TODO: Make postprocessor
# TODO: Make second preprocessor
# TODO: Make CLI
# TODO: Update the README
# TODO: Publish to PyPy