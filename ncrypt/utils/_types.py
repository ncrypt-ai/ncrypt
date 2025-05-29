from typing import Any

from cv2 import Mat, UMat
from numpy import dtype, floating, integer, ndarray

Cv2Image = UMat | Mat | ndarray[Any | dtype[integer | floating]]
