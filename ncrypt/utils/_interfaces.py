from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ncrypt.pdf import PDFFile


class TextRegion(ABC):
    @property
    @abstractmethod
    def page_num(self) -> int:
        pass

    @property
    @abstractmethod
    def page_attr(self) -> str | None:
        pass

    @property
    @abstractmethod
    def y_top(self) -> float:
        pass

    @property
    @abstractmethod
    def y_bottom(self) -> float:
        pass

    @property
    @abstractmethod
    def x_left(self) -> float:
        pass

    @property
    @abstractmethod
    def x_right(self) -> float:
        pass

    @property
    @abstractmethod
    def bounding_boxes(self) -> list[tuple[float, float, float, float]]:
        pass

    @abstractmethod
    def set_text(self, text: str) -> None:
        pass

    @abstractmethod
    def get_text(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class OCRModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        pass

    @property
    @abstractmethod
    def client_specs(self) -> tuple[dict[str, str], str]:
        pass

    @property
    @abstractmethod
    def model_specs(self) -> dict[str, str]:
        pass

    @abstractmethod
    def submit_job(self, file: "PDFFile", text_regions: list[list[TextRegion]], key: str) -> dict[str, int | list[str] | list[list[str]]]:
        pass

    @abstractmethod
    def get_job_status(self, file: "PDFFile", text_regions: list[list[TextRegion]], key: str) -> tuple["PDFFile", dict[str, str]]:
        pass


class PreProcessor(ABC):
    @abstractmethod
    def load_pdf(self, file_path: str) -> Union["PDFFile", None]:
        pass

    @abstractmethod
    def process(self, file: "PDFFile") -> tuple["PDFFile", list[list[TextRegion]]]:
        pass
