from __future__ import annotations

from email.mime import image
from pathlib import Path
from typing import Any

from PIL import Image
from paddleocr import PaddleOCR

from models.base.base_model import BaseMeterModel
from models.registry import ModelRegistry
import math
import numpy as np


class PaddleOCRBackend(BaseMeterModel):
    """
    PaddleOCR backend for electricity-meter text recognition.

    Uses the same 7-region strategy as the LightOnOCR backend:
        - full_image
        - tile_1
        - tile_2
        - tile_3
        - tile_4
        - tile_5
        - tile_6
    """

    def __init__(self) -> None:

        super().__init__("paddleocr")

        self.ocr = None
        self.is_loaded = False

        # Same tiling configuration as LightOnOCR.
        self.tile_rows = 2
        self.tile_columns = 3
        self.tile_overlap = 0.45

    def load(self, model_path: Path | None = None) -> None:
        """
        Initialize PaddleOCR.

        PaddleOCR downloads/loads its own OCR models, so model_path
        is currently unused.
        """

        self.ocr = PaddleOCR(
            lang="en",
            device="gpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        self.is_loaded = True

    def unload(self) -> None:
        """
        Release PaddleOCR resources.
        """

        self.ocr = None
        self.is_loaded = False

    def _create_tiles(
        self,
        image: Image.Image,
    ) -> list[Image.Image]:
        """
        Create overlapping padded tiles.

        Strategy:
            - 2 rows x 3 columns
            - 35% overlap
            - 10% padding around every tile
            - Edge tiles are padded rather than clipped

        The padding is synthetic image space. It does not recover
        pixels outside the original image; overlap is responsible
        for ensuring text near boundaries appears in another tile.
        """

        width, height = image.size

        rows = self.tile_rows
        columns = self.tile_columns

        overlap = self.tile_overlap

        # ---------------------------------------------------------
        # Nominal tile dimensions.
        # ---------------------------------------------------------

        tile_width = math.ceil(
            width / (
                columns
                - overlap * (columns - 1)
            )
        )

        tile_height = math.ceil(
            height / (
                rows
                - overlap * (rows - 1)
            )
        )

        # ---------------------------------------------------------
        # Distance between tile origins.
        # ---------------------------------------------------------

        step_x = int(
            tile_width * (1.0 - overlap)
        )

        step_y = int(
            tile_height * (1.0 - overlap)
        )

        # ---------------------------------------------------------
        # Proportional padding.
        # ---------------------------------------------------------

        padding_x = max(
            16,
            int(tile_width * 0.10),
        )

        padding_y = max(
            16,
            int(tile_height * 0.10),
        )

        tiles: list[Image.Image] = []

        for row in range(rows):

            for column in range(columns):

                # -------------------------------------------------
                # Desired crop position.
                # -------------------------------------------------

                x1 = column * step_x
                y1 = row * step_y

                x2 = x1 + tile_width
                y2 = y1 + tile_height

                # -------------------------------------------------
                # Shift the final tile so that it reaches the
                # image boundary instead of leaving uncovered
                # space.
                # -------------------------------------------------

                if column == columns - 1:
                    x2 = width
                    x1 = max(
                        0,
                        x2 - tile_width,
                    )

                if row == rows - 1:
                    y2 = height
                    y1 = max(
                        0,
                        y2 - tile_height,
                    )

                # -------------------------------------------------
                # Actual pixels available inside the image.
                # -------------------------------------------------

                crop_x1 = max(
                    0,
                    x1,
                )

                crop_y1 = max(
                    0,
                    y1,
                )

                crop_x2 = min(
                    width,
                    x2,
                )

                crop_y2 = min(
                    height,
                    y2,
                )

                crop = image.crop(
                    (
                        crop_x1,
                        crop_y1,
                        crop_x2,
                        crop_y2,
                    )
                )

                # -------------------------------------------------
                # Create padded canvas.
                # -------------------------------------------------

                padded_width = (
                    crop.width
                    + padding_x * 2
                )

                padded_height = (
                    crop.height
                    + padding_y * 2
                )

                padded = Image.new(
                    "RGB",
                    (
                        padded_width,
                        padded_height,
                    ),
                    "white",
                )

                padded.paste(
                    crop,
                    (
                        padding_x,
                        padding_y,
                    ),
                )

                tiles.append(
                    padded
                )

        return tiles

    def preprocess(
        self,
        image_path: Path,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Load image and generate the 7 regions.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        tiles = self._create_tiles(
            image
        )

        return {
            "image": image,
            "tiles": tiles,
        }

    def _run_ocr(
        self,
        image: Image.Image,
    ) -> str:
        """
        Run PaddleOCR on a single image region.

        PaddleOCR 3.x expects a numpy.ndarray or image path,
        not a PIL Image.
        """

        # ---------------------------------------------------------
        # FORCE RESIZE TO 1600px FOR SPEED
        # ---------------------------------------------------------
        max_size = 2400
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # ---------------------------------------------------------

        if self.ocr is None:
            raise RuntimeError(
                "PaddleOCR model is not loaded."
            )

        # PIL RGB -> NumPy RGB
        image_array = np.asarray(
            image.convert("RGB")
        )

        result = self.ocr.predict(
            input=image_array,
        )

        lines: list[str] = []

        for page in result:

            if page is None:
                continue

            data = page.json

            if callable(data):
                data = data()

            if not isinstance(data, dict):
                continue

            res = data.get(
                "res",
                data,
            )

            if not isinstance(res, dict):
                continue

            texts = res.get(
                "rec_texts",
                [],
            )

            scores = res.get(
                "rec_scores",
                [],
            )

            for index, text in enumerate(
                texts
            ):

                if not text:
                    continue

                text = str(text).strip()

                if not text:
                    continue

                # Keep reasonably confident OCR.
                # Lowered from 0.30 to 0.15 so that low-confidence
                # tile reads (e.g. serial near tile boundary) still
                # reach the extractor + consolidator layer.
                if scores:

                    try:
                        score = float(
                            scores[index]
                        )

                        if score < 0.15:
                            continue

                    except (
                        ValueError,
                        TypeError,
                        IndexError,
                    ):
                        pass

                lines.append(text)

        return "\n".join(lines)

    def predict(
        self,
        processed_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run PaddleOCR on the full image and all 6 tiles.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "PaddleOCR model is not loaded."
            )

        image = processed_input[
            "image"
        ]

        tiles = processed_input[
            "tiles"
        ]

        candidates: list[
            dict[str, str]
        ] = []

        # Full image.
        full_output = self._run_ocr(
            image
        )

        candidates.append(
            {
                "region": "full_image",
                "output": full_output,
            }
        )

        # Six tiles.
        for index, tile in enumerate(
            tiles
        ):

            output = self._run_ocr(
                tile
            )

            candidates.append(
                {
                    "region":
                        f"tile_{index + 1}",
                    "output": output,
                }
            )

        return {
            "candidates": candidates
        }

    def postprocess(
        self,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return raw OCR regions.

        Serial/IMEI extraction is intentionally kept outside
        the backend so PaddleOCR and LightOnOCR can use the
        same deterministic extraction/consolidation layer.
        """

        regions = []

        for candidate in prediction[
            "candidates"
        ]:

            regions.append(
                {
                    "region":
                        candidate["region"],
                    "raw_output":
                        candidate["output"],
                }
            )

        return {
            "model": self.model_name,
            "regions": regions,
        }

    def get_model_info(
        self,
    ) -> dict[str, Any]:

        return {
            "name": self.model_name,
            "device": "cpu",
            "loaded": self.is_loaded,
            "tile_rows":
                self.tile_rows,
            "tile_columns":
                self.tile_columns,
            "tile_overlap":
                self.tile_overlap,
        }


ModelRegistry.register(
    "paddleocr",
    PaddleOCRBackend,
)