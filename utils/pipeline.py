from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.ocr_extractor    import UniversalOCRExtractor
from utils.ocr_consolidator import UniversalOCRConsolidator
from models.backends.paddleocr.paddleocr_backend import PaddleOCRBackend


class MeterSerialPipeline:
    """
    End-to-end pipeline:
        image_path  →  {serial_number, imei}

    Wraps backend, extractor, and consolidator so callers
    (CLI, API, Colab) never need to know about the internals.
    Empty string means "not found".
    """

    def __init__(
        self,
        backend: PaddleOCRBackend | None = None,
        extractor: UniversalOCRExtractor | None = None,
        consolidator: UniversalOCRConsolidator | None = None,
    ) -> None:
        self.backend      = backend      or PaddleOCRBackend()
        self.extractor    = extractor    or UniversalOCRExtractor()
        self.consolidator = consolidator or UniversalOCRConsolidator()

        if not self.backend.is_loaded:
            self.backend.load()

    def run(self, image_path: Path | str) -> dict[str, Any]:
        """
        Run the full pipeline on a single image.

        Returns:
            {
              "serial_number": str,   # "" if not found
              "imei":          str,   # "" if not found
            }
        """
        image_path = Path(image_path)

        pre  = self.backend.preprocess(image_path)
        pred = self.backend.predict(pre)
        post = self.backend.postprocess(pred)

        # region_outputs values are dicts: {region, raw_output, scored_lines}
        region_outputs = {r["region"]: r for r in post["regions"]}

        # Per-region extraction, then cross-region vote
        per_region_extracts = {
            reg: self.extractor.extract(r["raw_output"])
            for reg, r in region_outputs.items()
        }
        consolidated = self.consolidator.consolidate(per_region_extracts)

        return {
            "serial_number": consolidated.get("serial_number", "") or "",
            "imei":          consolidated.get("imei", "")          or "",
        }