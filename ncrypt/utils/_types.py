from typing import Any, Dict, List, Literal, TypedDict

from cv2 import Mat, UMat
from numpy import dtype, floating, integer, ndarray

Cv2Image = UMat | Mat | ndarray[Any | dtype[integer | floating]]

class _Result(TypedDict):
    status: Literal["complete", "incomplete"]
    output: str
    
class JobResults(TypedDict):
    num_jobs: int
    num_completed_jobs: int
    status: int
    pages: Dict[str, Dict[str, _Result]]
