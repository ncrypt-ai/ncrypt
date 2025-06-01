import math

import cv2
import fitz
import numpy as np

from ncrypt.line import Line
from ncrypt.pdf import PDFFile
from ncrypt.utils import Cv2Image, PreProcessor


class DefaultPreprocessor(PreProcessor):
    def __init__(self, dpi: int = 300) -> None:
        """
        Initializes the DefaultPreprocessor instance.
        """
        self.dpi = dpi

    def load_pdf(self, file_path: str) -> PDFFile | None:
        pdf = fitz.open(file_path)
        pages = []

        for page_number in range(len(pdf)):
            page = pdf.load_page(page_number)

            pix = page.get_pixmap(dpi=self.dpi)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                (pix.height, pix.width, pix.n)
            )

            # Convert RGBA to BGR if needed
            if pix.n == 4:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

            else:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            pages.append(img_bgr)

        return None if not pages else PDFFile(pages)

    def process(self, file: PDFFile) -> tuple[PDFFile, list[list[Line]]]:
        text_regions = []

        for idx in range(file.num_pages):
            page = file.get_page(idx)
            greyscale = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

            brightened = self.__enhance_contrast(greyscale)
            blurred = cv2.medianBlur(brightened, 5)

            smoothed = cv2.morphologyEx(
                blurred, cv2.MORPH_CLOSE, np.ones((2, 2)), iterations=1
            )
            smoothed = cv2.morphologyEx(
                smoothed, cv2.MORPH_ERODE, np.ones((3, 2)), iterations=1
            )

            gaps = np.zeros_like(smoothed)
            dynamic_mask = self.__dynamic_denoise(smoothed, hi=255)
            dynamic_denoised = cv2.bitwise_and(255 - smoothed, dynamic_mask)

            thresh_1 = cv2.adaptiveThreshold(
                dynamic_denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=61,
                C=10,
            )
            thresh_2 = cv2.ximgproc.niBlackThreshold(
                dynamic_denoised,
                255,
                cv2.THRESH_BINARY,
                blockSize=121,
                k=-1.1,
                binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
            )
            _, thresh_3 = cv2.threshold(dynamic_denoised, 25, 230, cv2.THRESH_BINARY)

            binary_mask = cv2.bitwise_and(thresh_1, thresh_2)
            binarized = cv2.bitwise_and(binary_mask, thresh_3)

            inverted = 255 - binarized
            background_mask, text_mask = self.__find_text_regions(binarized)

            file.add_transformation(binarized, "binarized")
            file.add_transformation(inverted, "inverted")
            file.add_transformation(cv2.bitwise_or(gaps, background_mask), "background")
            file.add_transformation(cv2.bitwise_and(binarized, text_mask), "foreground")

            lines = self.__find_entities(
                cv2.bitwise_and(binarized, text_mask), background_mask, gaps, idx
            )
            text_regions.append(lines)
            file.text_regions.append(lines)

        return file, text_regions

    @staticmethod
    def __enhance_contrast(img: Cv2Image) -> Cv2Image:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(img)

        return clahe_img

    @staticmethod
    def __static_denoise(
        img: Cv2Image, hi: int = 255, percentile: int = 90, auto: bool = False
    ) -> Cv2Image:
        height, width = img.shape
        cols = []
        rows = []

        vertical_hist = np.sum(img, axis=0) / hi
        horizontal_hist = np.sum(img, axis=1) / hi
        vertical_thresh = (
            np.percentile(vertical_hist, percentile) if not auto else (1 / 3) * width
        )
        horizontal_thresh = (
            np.percentile(horizontal_hist, percentile) if not auto else (1 / 3) * height
        )

        horizontal_mask = np.zeros_like(img)
        vetical_mask = np.zeros_like(img)

        for i in range(width):
            if vertical_hist[i] > vertical_thresh:
                vetical_mask[:, i] = hi
                cols.append(i)

        for i in range(height):
            if horizontal_hist[i] >= horizontal_thresh:
                horizontal_mask[i, :] = hi
                rows.append(i)

        combined_mask = cv2.bitwise_and(vetical_mask, horizontal_mask)

        return combined_mask, rows, cols

    @staticmethod
    def __dynamic_denoise(img: Cv2Image, hi: int = 255) -> Cv2Image:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img, connectivity=4
        )
        height_thresh = np.median(stats[1:, 3])
        mask = np.ones_like(img) * hi

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            elongation = min(w, h) / max(w, h)
            density = (np.sum(labels[labels == i]) / i) / area

            if (
                h <= math.ceil(height_thresh / 4)
                or h <= 2
                or area <= 4
                or elongation <= 0.08
                or density <= 0.08
            ):
                mask[y : y + h + 1, x : x + w + 1] = 0

        return mask

    @staticmethod
    def __arlsa(
        img: Cv2Image,
        gap_size: int,
        length_thresh: float = 1.5,
        height_thresh: float = 3.5,
        overlap_thresh: float = 0.4,
        hi: int = 255,
    ) -> tuple[Cv2Image, Cv2Image]:
        """
        https://users.iit.demokritos.gr/~bgat/IMAVIS_segm.pdf
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        img = 255 - img
        row_mask = np.ones_like(img) * hi

        for row in range(img.shape[0]):
            last_component = None
            last_component_idx = 0

            for col in range(img.shape[1]):
                if labels[row][col] != 0:
                    next_component = labels[row][col]
                    next_component_idx = col

                    if last_component_idx != next_component_idx:
                        seq_len = next_component_idx - last_component_idx

                        if last_component == next_component:
                            min_height = stats[next_component, cv2.CC_STAT_HEIGHT]

                            if seq_len <= math.ceil(length_thresh * min_height):
                                row_mask[
                                    row, last_component_idx : next_component_idx + 1
                                ] = 0

                        elif last_component and last_component != next_component:
                            y1 = stats[last_component, cv2.CC_STAT_TOP]
                            h1 = stats[last_component, cv2.CC_STAT_HEIGHT]

                            y2 = stats[next_component, cv2.CC_STAT_TOP]
                            h2 = stats[next_component, cv2.CC_STAT_HEIGHT]
                            min_height = min(h1, h2)

                            height_ratio = max(h1, h2) / min_height
                            overlap = abs(max(y1, y2) - min(y1 + h1, y2 + h2))

                            if (
                                seq_len <= math.ceil(length_thresh * gap_size)
                                and height_ratio <= height_thresh
                                and overlap >= math.ceil(overlap_thresh * min_height)
                            ):
                                if 0 < row < img.shape[0] - 1:
                                    has_third_component = False

                                    for j in range(seq_len):
                                        up = labels[row - 1][last_component_idx + j]
                                        down = labels[row + 1][last_component_idx + j]

                                        if (
                                            up != 0
                                            and up != last_component
                                            and up != next_component
                                        ):
                                            has_third_component = True
                                            break

                                        if (
                                            down != 0
                                            and down != last_component
                                            and down != next_component
                                        ):
                                            has_third_component = True
                                            break

                                    if not has_third_component:
                                        row_mask[
                                            row,
                                            last_component_idx : next_component_idx + 1,
                                        ] = 0

                        last_component = next_component
                        last_component_idx = next_component_idx

        return row_mask

    @staticmethod
    def __find_text_regions(
        img: Cv2Image, hi: int = 255, filter_rects: bool = True
    ) -> Cv2Image:
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        text_mask = np.zeros_like(img)
        text_regions = []
        heights = []

        for contour in contours:
            _, _, _, h = cv2.boundingRect(contour)
            heights.append(h)

        median = np.median(heights)
        mad = np.median(np.abs(heights - median))

        hi_thresh = np.percentile(heights, 99)
        lo_thresh = np.percentile(heights, 5)

        if filter_rects:
            for i in range(len(contours)):
                contour = contours[i]
                height = heights[i]

                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

                contour_mask = np.zeros_like(img)
                cv2.drawContours(contour_mask, [approx], -1, hi, cv2.FILLED)
                density = cv2.mean(img, mask=contour_mask)[0]

                if density > 127 and (height >= lo_thresh or abs(height - median) < 2 * mad) and (height <= hi_thresh or abs(height - median) < 2 * mad):
                    text_regions.append(contour)

        for contour in text_regions:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                text_mask, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED
            )

        background_mask = cv2.bitwise_and(img, 255 - text_mask)

        return background_mask, text_mask

    @staticmethod
    def __compute_median_gap(components: list[tuple[float, float, float, float, float]]):
        """
        Assumes that each of the connected components are on the same line and are being read left to right.
        """
        components = sorted(components, key=lambda x: (x[0] + x[2]) / 2)
        gaps = []

        for i in range(1, len(components)):
            x1, _, _, _, _ = components[i]
            x2, _, w2, _, _ = components[i - 1]

            gaps.append(x1 - (x2 + w2))

        return 0 if not gaps else np.median(gaps)

    def __find_entities(
        self,
        img: Cv2Image,
        background_mask: Cv2Image,
        gap_mask: Cv2Image,
        page_num: int
    ):
        """
        https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33418.pdf
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3db9a4d4fbc6967b2bc80f47e764f44af39c6867
        """
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img)
        dead_space = cv2.bitwise_or(background_mask, gap_mask)
        sorted_labels = sorted(
            range(1, num_labels), key=lambda l: (stats[l, 0], stats[l, 1])
        )

        rows = {}
        row_idx = 0
        alpha = 0.6

        for label in sorted_labels:
            x, y, w, h, _ = stats[label]

            # Find the existing row with the most vertical overlap if one exists
            max_overlap = 0
            shift = 0
            overlapped_row = None

            for row in rows:
                running_alpha, y_top, y_bottom, x_left, x_right, label_nums = rows[row]
                vertical_overlap = max(
                    0,
                    min(y_bottom, y + h + running_alpha)
                    - max(y_top, y + running_alpha),
                )

                if vertical_overlap > max_overlap:
                    obstacles = cv2.countNonZero(
                        dead_space[min(y_top, y) : max(y_bottom, y + h), x_right:x]
                    )

                    if obstacles == 0:
                        max_overlap = vertical_overlap
                        overlapped_row = row
                        shift = (-1 if y_top > y else 1) * abs(h - vertical_overlap)

            if max_overlap == 0 and not overlapped_row:
                rows[row_idx] = (0, y, y + h, x, x + w, [label])

            else:
                running_alpha, y_top, y_bottom, x_left, x_right, label_nums = rows[
                    overlapped_row
                ]
                running_alpha = alpha * running_alpha + (1 - alpha) * shift
                y_top = min(y_top, y)
                y_bottom = max(y_bottom, y + h)
                x_right = max(x_right, x + w)
                x_left = min(x_left, x)
                label_nums = label_nums + [label]

                rows[overlapped_row] = (
                    running_alpha,
                    y_top,
                    y_bottom,
                    x_left,
                    x_right,
                    label_nums,
                )

            row_idx += 1

        lines: list[Line] = []

        for row in rows:
            running_alpha, y_top, y_bottom, x_left, x_right, label_nums = rows[row]
            window = img[y_top:y_bottom, x_left:x_right]
            bboxes = [stats[label] for label in label_nums]

            median_gap = self.__compute_median_gap(bboxes)
            smoothed = self.__arlsa(window, median_gap, 2.5)

            _, _, word_bboxes, _ = cv2.connectedComponentsWithStats(255 - smoothed)
            word_bboxes = [(x_left + x, y_top + y, w, h) for x, y, w, h, _ in word_bboxes[1:]]

            lines.append(
                Line(
                    running_alpha,
                    y_top,
                    y_bottom,
                    x_left,
                    x_right,
                    page=page_num,
                    page_attr="foreground",
                    bounding_boxes=word_bboxes,
                )
            )

        return lines
