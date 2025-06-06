import io
import os
import pickle
from typing import Dict, List

import fitz
from PIL import Image

from ncrypt.utils import Cv2Image
from ncrypt.utils._interfaces import TextRegion

from .line import Line


class PDFFile:
    def __init__(self, pages: list[Cv2Image]) -> None:
        """
        Initializes the PDFFile instance.

        :param pages: A list of numpy ndarrays, corresponding to each of the pages of the PDF.
        """
        self.pages: list[Cv2Image] = pages
        self.transformations: dict[str, list[Cv2Image]] = {}
        self.text_regions: list[list[TextRegion]] = []

        self.page_ids: list[str] = []
        self.job_ids: list[list[str]] = []

    @property
    def preprocessed(self) -> bool:
        """
        Returns whether or not the PDF has been preprocessed.
        """
        for k in self.transformations.keys():
            if len(self.transformations[k]) != len(self.pages) and len(self.pages) != len(self.text_regions):
                return False

        return True

    @property
    def submitted(self) -> bool:
        """
        Returns whether or not the PDF has been submitted for OCR.
        """
        return len(self.page_ids) == len(self.pages)

    @property
    def num_pages(self) -> int:
        """
        Returns the number of pages in the PDF.
        """
        return len(self.pages)

    def add_page(self, img: Cv2Image) -> None:
        """
        Saves the Cv2Image representation of a single page.

        :param img: A CV2Image representing a single page of the PDFFile.
        :return: None
        """
        self.pages.append(img)

    def add_transformation(self, img: Cv2Image, transformation: str) -> None:
        """
        Saves the outputs of a transformation on a single page.

        :param img: A CV2Image representing a preprocessing step applied to a single page of the PDFFile.
        :param transformation: Name of the transformation that was applied.
        :return: None
        """
        if transformation not in self.transformations:
            self.transformations[transformation] = [img]

        else:
            self.transformations[transformation].append(img)

    def add_text_regions(self, text_regions: List[TextRegion]) -> None:
        """
        Sets each of the text regions that are present in each page.

        :param text_regions: A TextRegion object representing all of the text regions on a page.
        :return: None
        """
        self.text_regions.append(text_regions)

    def add_ids(self, page_id: str, job_ids: list[str]) -> None:
        """
        Sets the page and job IDs generated by the ncrypt service.

        :param page_id: A unique identifier for a single page of the pdf.
        :param job_ids: A unique set of identifiers for each of the batch jobs within a single page.
        :return:
        """
        if not page_id:
            raise ValueError("There must be a job ID specified.")

        if not job_ids:
            raise ValueError("There must be at least one job ID for the page.")

        self.page_ids.append(page_id)
        self.job_ids.append(job_ids)

    def get_page(self, idx: int) -> Cv2Image | None:
        """
        Fetches the page at the given index.

        :param idx: Index of the page to fetch. Zero indexed.
        :return: The page at the given index.
        """
        if 0 <= idx < self.num_pages:
            return self.pages[idx].copy()

        else:
            raise IndexError("Page index out of range.")

    def get_transformation(self, idx: int, transformation: str) -> Cv2Image | None:
        """
        Fetches the name transformation at the given index.

        :param idx: Index of the page to fetch. Zero indexed.
        :param transformation: Name of the transformation to return.
        :return: The page at the given index.
        """
        if not self.preprocessed:
            raise LookupError("The PDF has not been preprocessed.")

        if idx and (idx < 0 or idx >= len(self.pages)):
            raise IndexError("Page index out of range.")

        if transformation not in self.transformations:
            raise KeyError("No such transformation exists.")

        return self.transformations[transformation][idx]
    
    def get_text_regions(self, idx: int | None = None) -> List[List[TextRegion]]:
        """
        Fetches thelist of text regions at the given index if one is specified, else all text_regions.

        :param idx: Index of the IDs to fetch. Zero indexed.
        :return: A list of lists containing TextRegion objects.
        """
        if not self.submitted:
            raise LookupError("The PDF has not been submitted to the OCR endpoint.")

        if idx and (idx < 0 or idx >= len(self.pages)):
            raise IndexError("Page index out of range.")
        
        return [self.text_regions[idx]] if idx else self.text_regions

    def get_page_ids(self, idx: int | None = None) -> Dict[str, List[str] | List[List[str]]] | None:
        """
        Fetches the page and job IDs at the given index if one is specified, else all IDs.

        :param idx: Index of the IDs to fetch. Zero indexed.
        :return: An object with the keys 'page_ids' and 'job_ids'.
        """
        if not self.submitted:
            raise LookupError("The PDF has not been submitted to the OCR endpoint.")

        if idx and (idx < 0 or idx >= len(self.pages)):
            raise IndexError("Page index out of range.")

        page_ids = [self.page_ids[idx]] if idx else self.page_ids
        job_ids = [self.job_ids[idx]] if idx else self.job_ids

        return {
            "page_ids": page_ids,
            "job_ids": job_ids
        }

    def save(self, filename: str, out_dir: str) -> None:
        """
        Save multiple numpy arrays as a multi-page PDF file.

        :param filename: The name to save the pdf as.
        :param out_dir: The path to save the pdf at.
        """
        if len(filename) < 4 or filename[-4:] != ".pdf":
            raise ValueError("The filename must end with '.pdf'.")

        doc = fitz.open()

        for array in self.pages:
            if len(array.shape) == 2:  # Grayscale
                image = Image.fromarray(array).convert("RGB")

            elif len(array.shape) == 3 and array.shape[2] in [3, 4]:  # RGB or RGBA
                image = Image.fromarray(array[:, :, :3])

            else:
                raise ValueError("Unsupported array shape for image conversion.")

            # Save image to in-memory buffer
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)

            # Open image as a PDF and insert it
            img_pdf = fitz.open("png", img_buffer.read())
            doc.insert_pdf(img_pdf)

        doc.save(os.path.join(out_dir, filename))
        doc.close()

    def serialize(self, filename: str, out_dir: str) -> None:
        """
        Serialize the PDFFile object.

        :param filename: The name to save the pdf as.
        :param out_dir: The path to save the pdf at.
        """
        with open(os.path.join(out_dir, filename), "wb") as file:
            pickle.dump(self, file)
