from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from paddleocr import PaddleOCR

import paddle

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

    Device selection is automatic:
        - If a GPU is attached AND Paddle was compiled with CUDA → use GPU
        - Otherwise → use CPU
    This means the same code works in GPU-enabled and CPU-only
    Colab runtimes without manual edits.

    Confidence handling:
        - A global floor (self.min_ocr_score = 0.15) drops near-garbage
          OCR lines before they reach downstream code.
        - A stricter IMEI floor (self.imei_min_confidence = 0.70) is
          carried through postprocess() so the extractor can apply it
          only to IMEI candidates. Serial extraction keeps using the
          permissive global floor, because serial has additional
          safeguards (pattern gate + multi-region consensus).
    """

    def __init__(self) -> None:

        super().__init__("paddleocr")

        self.ocr = None
        self.is_loaded = False
        self._device = "cpu"   # actual device used after load()

        # Same tiling configuration as LightOnOCR.
        self.tile_rows = 2
        self.tile_columns = 3
        self.tile_overlap = 0.45

        # ---------------------------------------------------------
        # NEW: per-field confidence floors.
        #
        # self.min_ocr_score — global floor applied inside _run_ocr.
        #   Kept low so partial boundary reads survive for the
        #   multi-region consolidator to stitch together.
        #
        # self.imei_min_confidence — stricter floor for IMEI only.
        #   Not applied here; carried through postprocess() and
        #   enforced by UniversalOCRExtractor._imei_candidates().
        # ---------------------------------------------------------
        self.min_ocr_score = 0.15
        self.imei_min_confidence = 0.70

    def load(self, model_path: Path | None = None) -> None:
        """
        Initialize PaddleOCR.

        Auto-detects whether a GPU is available. PaddlePaddle's
        `is_compiled_with_cuda()` alone is not enough — the wheel
        might be CUDA-enabled but the runtime might have no GPU
        attached. `device_count() > 0` confirms an actual GPU.

        PaddleOCR downloads/loads its own OCR models, so model_path
        is currently unused.
        """

        # ---- Auto-detect device ----
        device = "cpu"
        try:
            if (
                paddle.device.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0
            ):
                device = "gpu"
        except Exception:
            device = "cpu"

        self._device = device

        self.ocr = PaddleOCR(
            lang="en",
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        self.is_loaded = True

        print(f"[PaddleOCRBackend] Loaded on device: {device}")

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
            - 45% overlap
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
    ) -> list[tuple[str, float]]:
        """
        Run PaddleOCR on a single image region.

        Returns a list of (text, score) tuples instead of a
        newline-joined string, so that downstream code can apply
        field-specific confidence floors (e.g. stricter for IMEI).

        PaddleOCR 3.x expects a numpy.ndarray or image path,
        not a PIL Image.
        """

        # ---------------------------------------------------------
        # Cap image size to keep OCR fast and memory-safe.
        # ---------------------------------------------------------
        max_size = 2400
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )
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

        # ---------------------------------------------------------
        # CHANGED: list of (text, score) tuples instead of bare
        # strings. Score is preserved so the extractor can apply
        # field-specific floors.
        # ---------------------------------------------------------
        lines: list[tuple[str, float]] = []

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
                # Global floor is intentionally low — tile reads
                # near boundaries often score lower, and the
                # multi-region consolidator is designed to stitch
                # partial reads together.
                score = 1.0
                if scores:
                    try:
                        score = float(scores[index])
                    except (
                        ValueError,
                        TypeError,
                        IndexError,
                    ):
                        score = 1.0

                if score < self.min_ocr_score:
                    continue

                lines.append((text, score))

        return lines

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
            dict[str, Any]
        ] = []

        # Full image.
        full_output = self._run_ocr(
            image
        )

        candidates.append(
            {
                "region": "full_image",
                "output": full_output,   # list[(text, score)]
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
                    "output": output,    # list[(text, score)]
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
        Return raw OCR regions plus scored lines.

        Serial/IMEI extraction is intentionally kept outside
        the backend so PaddleOCR and LightOnOCR can use the
        same deterministic extraction/consolidation layer.

        Each region now carries THREE pieces of information:
            - raw_output    : str — backward-compatible flat text
            - scored_lines  : list[(text, score)] — for field gates
            - And the top-level result carries imei_min_confidence
              so the extractor knows which floor to enforce.
        """

        regions = []

        for candidate in prediction[
            "candidates"
        ]:

            scored_lines = candidate["output"]   # list[(text, score)]
            flat_text = "\n".join(
                text for text, _score in scored_lines
            )

            regions.append(
                {
                    "region":
                        candidate["region"],
                    "raw_output":
                        flat_text,
                    "scored_lines":
                        scored_lines,
                }
            )

        return {
            "model": self.model_name,
            "regions": regions,
            "imei_min_confidence": self.imei_min_confidence,
        }

    def get_model_info(
        self,
    ) -> dict[str, Any]:

        return {
            "name": self.model_name,
            "device": self._device,   # actual device, not hardcoded
            "loaded": self.is_loaded,
            "tile_rows":
                self.tile_rows,
            "tile_columns":
                self.tile_columns,
            "tile_overlap":
                self.tile_overlap,
            "min_ocr_score":
                self.min_ocr_score,
            "imei_min_confidence":
                self.imei_min_confidence,
        }


ModelRegistry.register(
    "paddleocr",
    PaddleOCRBackend,
)