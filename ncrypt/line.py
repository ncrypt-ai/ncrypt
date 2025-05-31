from typing import List, Tuple, Union

from ncrypt.utils import TextRegion, Cv2Image


class Line(TextRegion):
    def __init__(
        self,
        alpha: float,
        y_top: float,
        y_bottom: float,
        x_left: float,
        x_right: float,
        page: int,
        page_attr: Union[str, None],
        bounding_boxes: List[Tuple[float, float, float, float]] = [],
    ) -> None:
        self.alpha = alpha
        self.y1 = y_top
        self.y2 = y_bottom
        self.x1 = x_left
        self.x2 = x_right
        self.width = x_right - x_left
        self.height = y_bottom - y_top
        self.page = page
        self.attr = page_attr
        self.bboxes = bounding_boxes if len(bounding_boxes) == 1 else self.__vertical_merge(bounding_boxes)
        self.text = ""

    @property
    def y_top(self) -> float:
        return self.y1

    @property
    def y_bottom(self) -> float:
        return self.y2

    @property
    def x_left(self) -> float:
        return self.x1

    @property
    def x_right(self) -> float:
        return self.x2

    @property
    def bounding_boxes(self) -> List[Tuple[float, float, float, float]]:
        if self.bboxes:
            return self.bboxes

        return [(self.x_left, self.y_top, self.width, self.height)]

    @property
    def page_num(self) -> int:
        return self.page

    @property
    def page_attr(self) -> int:
        return self.attr

    @staticmethod
    def __vertical_merge(
        bounding_boxes
    ) -> List[Tuple[float, float, float, float]]:
        sorted_boxes = sorted(bounding_boxes, key=lambda x: x[0] + x[3] // 2)
        combined = []

        for x1, y1, w1, h1 in sorted_boxes:
            if combined:
                x2, y2, w2, h2 = combined.pop(-1)
                overlap = max(
                    0,
                    min(x1 + w1, x2 + w2) - max(x1, x2),
                )

                if overlap >= 0.5 * min(w1, w2):
                    left = min(x1, x2)
                    right = max(x1 + w1, x2 + w2)
                    top = min(y1, y2)
                    bottom = max(y1 + h1, y2 + h2)

                    combined.append((left, top, right - left, bottom - top))

                else:
                    combined.append((x2, y2, w2, h2))
                    combined.append((x1, y1, w1, h1))

            else:
                combined.append((x1, y1, w1, h1))

        return sorted(combined, key=lambda x: x[0] + x[3] // 2)

    def set_text(self, text: str) -> None:
        self.text = text

    def get_text(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text
