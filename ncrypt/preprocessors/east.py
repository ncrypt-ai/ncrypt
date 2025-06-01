
import cv2
import fitz
import numpy as np
from imutils.object_detection import non_max_suppression

from ncrypt.line import Line
from ncrypt.pdf import PDFFile
from ncrypt.utils import PreProcessor


class EastPreprocessor(PreProcessor):
    def __init__(self, dpi: int = 300, min_confidence: float = 0.3) -> None:
        """
        Initializes the EastPreprocessor instance.
        """
        self.dpi = dpi
        self.model_dir = "ncrypt/static/frozen_east_text_detection.pb"
        self.min_confidence = min_confidence

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

            # Resize to make dimensions multiples of 32
            height, width = img_bgr.shape[:2]
            new_height = ((height + 31) // 32) * 32
            new_width = ((width + 31) // 32) * 32

            if new_height != height or new_width != width:
                img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            pages.append(img_bgr)

        return None if not pages else PDFFile(pages)

    def process(self, file: PDFFile) -> tuple[PDFFile, list[list[Line]]]:
        """
        https://github.com/gifflet/opencv-text-detection/blob/master/text_detection.py
        """
        east = cv2.dnn.readNet(self.model_dir)
        layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]
        text_regions: list[list[Line]] = []

        for idx in range(file.num_pages):
            page = file.get_page(idx)
            height, width, _ = page.shape

            blob = cv2.dnn.blobFromImage(page, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
            east.setInput(blob)
            (scores, geometry) = east.forward(layer_names)

            (num_rows, num_cols) = scores.shape[2:4]
            rects = []
            confidences = []

            for y in range(0, num_rows):
                scores_data = scores[0, 0, y]
                x0 = geometry[0, 0, y]
                x1 = geometry[0, 1, y]
                x2 = geometry[0, 2, y]
                x3 = geometry[0, 3, y]
                angles_data = geometry[0, 4, y]

                for x in range(0, num_cols):
                    if scores_data[x] < self.min_confidence:
                        continue

                    # Compute the offset factor as our resulting feature maps will be 4x smaller than the input image
                    (offset_x, offset_y) = (x * 4.0, y * 4.0)

                    # Extract the rotation angle for the prediction and then compute the sin and cosine
                    angle = angles_data[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    # Use the geometry volume to derive the width and height of the bounding box
                    h = x0[x] + x2[x]
                    w = x1[x] + x3[x]

                    # Compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
                    end_x = int(offset_x + (cos * x1[x]) + (sin * x2[x]))
                    end_y = int(offset_y - (sin * x1[x]) + (cos * x2[x]))
                    start_x = int(end_x - w)
                    start_y = int(end_y - h)

                    # Add the bounding box coordinates and probability score to our respective lists
                    rects.append((start_x, start_y, end_x, end_y))
                    confidences.append(scores_data[x])

            # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
            bboxes = non_max_suppression(np.array(rects), probs=confidences)
            lines: list[Line] = []

            greyscale = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
            foreground = np.ones((width, height), dtype=np.uint8) * 255
            background = greyscale

            for x_left, y_top, x_right, y_bottom in bboxes:
                window = greyscale[y_top:y_bottom, x_left:x_right]
                binarized = cv2.adaptiveThreshold(
                    window,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=61,
                    C=10,
                )

                foreground[y_top:y_bottom, x_left:x_right] = binarized
                background[y_top:y_bottom, x_left:x_right] = np.ones_like(window, dtype=np.uint8) * 255
                lines.append(Line(0, y_top, y_bottom, x_left, x_right, page=idx, page_attr="foreground", bounding_boxes=[(x_left, y_top, x_right - x_left, y_bottom - y_top)]))

            file.add_transformation(background, "background")
            file.add_transformation(foreground, "foreground")
            text_regions.append(lines)

        return file, text_regions
