from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ncrypt.pdf import PDFFile


class TextRegion(ABC):
    @property
    @abstractmethod
    def page_num(self) -> int:
        pass

    @property
    @abstractmethod
    def page_attr(self) -> Union[str, None]:
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
    def bounding_boxes(self) -> List[Tuple[float, float, float, float]]:
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
    def client_specs(self) -> Tuple[Dict[str, str], str]:
        pass

    @property
    @abstractmethod
    def model_specs(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def submit_job(self, file: "PDFFile", text_regions: List[List[TextRegion]], key: str) -> Dict[str, Union[int, List[str], List[List[str]]]]:
        pass

    @abstractmethod
    def get_job_status(self, file: "PDFFile", text_regions: List[List[TextRegion]], key: str) -> Tuple["PDFFile", Dict[str, str]]:
        pass


class PreProcessor(ABC):
    @abstractmethod
    def load_pdf(self, file_path: str) -> Union["PDFFile", None]:
        pass

    @abstractmethod
    def process(self, file: "PDFFile") -> Tuple["PDFFile", List[List[TextRegion]]]:
        pass
