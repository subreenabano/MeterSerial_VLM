# importing libraries

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
)

from models.base.base_model import BaseMeterModel
from models.registry import ModelRegistry
from utils.ocr_extractor import UniversalOCRExtractor as LightOnOCRExtractor

class LightOnOCRBackend(BaseMeterModel):
    """
    LightOnOCR-2-1B backend for electricity meter OCR.
    """

    def __init__(self) -> None:

        super().__init__("lightonocr")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.device.type == "cuda":
            self.dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self.dtype = torch.float32

        self.model = None
        self.processor = None

        self.tile_rows = 2
        self.tile_columns = 3
        self.tile_overlap = 0.20

    def load(self, model_path: Path) -> None:
        """
        Load the locally stored LightOnOCR model.
        """

        self.processor = LightOnOcrProcessor.from_pretrained(
            model_path,
            fix_mistral_regex=True
        )

        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
        )

        self.model.eval()
        self.model.to(self.device)
        self.is_loaded = True

    def unload(self) -> None:
        """
        Release model resources.
        """

        self.model = None
        self.processor = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_tiles(
        self,
        image: Image.Image,
    ) -> list[Image.Image]:
        """
        Create generic overlapping image tiles.
        """

        width, height = image.size

        rows = self.tile_rows
        columns = self.tile_columns

        base_width = width / columns
        base_height = height / rows

        step_x = base_width * (1.0 - self.tile_overlap)
        step_y = base_height * (1.0 - self.tile_overlap)

        tile_width = min(
            width,
            int(base_width * (1.0 + self.tile_overlap)),
        )

        tile_height = min(
            height,
            int(base_height * (1.0 + self.tile_overlap)),
        )

        tiles: list[Image.Image] = []

        for row in range(rows):

            for column in range(columns):

                x = int(column * step_x)
                y = int(row * step_y)

                x = min(
                    x,
                    max(0, width - tile_width),
                )

                y = min(
                    y,
                    max(0, height - tile_height),
                )

                tile = image.crop(
                    (
                        x,
                        y,
                        x + tile_width,
                        y + tile_height,
                    )
                )

                tiles.append(tile)

        return tiles

    def _build_prompt(self) -> str:
        """
        Build a meter-specific OCR instruction.
        """

        return """
    Read the text visible on this electricity smart meter.

    This is a digital electricity meter. Your primary task is to accurately
    read the meter identification information printed on the meter body,
    especially the identification label or specification plate.

    PRIORITY 1 — METER SERIAL NUMBER

    Look carefully for the meter's serial number and its associated label.

    Common labels include:
    - Serial Number
    - Serial No
    - Serial No.
    - S/N
    - S. No.
    - S No
    - SL NO
    - SL. NO
    - SL NO.
    - Meter Number
    - Meter No
    - Meter No.
    - Meter ID
    - Device ID
    - Device No

    The value immediately associated with these labels is the meter serial
    number.

    PRIORITY 2 — IMEI NUMBER

    Look for the cellular communication IMEI printed on the meter.

    Common labels include:
    - IMEI
    - IMEI NO
    - IMEI NO.
    - IMEI Number
    - IMEI NUMBER

    The value associated with the IMEI label is the IMEI number.

    PRIORITY 3 — METER DATES

    Look for dates printed on the meter identification/specification plate.

    Pay attention to:
    - MFG
    - MFG.
    - Manufactured
    - Manufacturing Date
    - Manufacture Date
    - Dated
    - Date
    - Installation Date
    - Installed
    - Commissioning Date
    - Commissioned

    Read the date exactly as printed.

    IMPORTANT VISUAL INSTRUCTIONS

    The serial number and IMEI may be printed in small text on the meter
    identification plate.

    Carefully inspect:
    - identification plates
    - specification labels
    - printed text near "SL NO" or "S/N"
    - printed text near "IMEI"
    - text near manufacturer information
    - small text containing letters and numbers

    Prioritize identification information over the large digital meter
    display.

    CHARACTER ACCURACY

    Read every visible character exactly as it appears.

    Pay special attention to characters that are easily confused:
    - O and 0
    - I and 1
    - S and 5
    - B and 8
    - G and 6
    - Z and 2

    Do not automatically correct or guess an unclear character.

    Do not invent missing characters.

    IMPORTANT — DO NOT CONFUSE THESE VALUES

    Do NOT treat the following as the meter serial number unless they are
    explicitly associated with a serial-number label:

    - IMEI numbers
    - meter readings
    - electricity consumption values
    - voltage
    - current
    - frequency
    - latitude
    - longitude
    - dates
    - manufacturing numbers
    - model numbers
    - part numbers
    - firmware numbers
    - QR codes
    - barcode numbers
    - manufacturer names
    - product names

    Likewise, do not treat the serial number as an IMEI unless it is
    explicitly associated with an IMEI label.

    OUTPUT

    Transcribe the relevant visible meter identification text.

    If the identification information is visible, return the text exactly
    as it appears on the meter.

    If no relevant identification information is visible, return:

    NOT_FOUND

    Do not invent information.
    """.strip()

    def _run_single_image(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Run LightOnOCR on one image region.
        """

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[-1]

        generated_ids = output_ids[
            0,
            input_length:,
        ]

        output_text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return output_text.strip()

    def preprocess(
        self,
        image_path: Path,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Prepare full image and generic tiles.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        tiles = self._create_tiles(image)

        return {
            "image": image,
            "tiles": tiles,
            "prompt": prompt or self._build_prompt(),
        }

    def predict(
        self,
        processed_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run LightOnOCR on the full image and all tiles.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "LightOnOCR model is not loaded."
            )

        image = processed_input["image"]
        tiles = processed_input["tiles"]
        prompt = processed_input["prompt"]

        candidates: list[dict[str, str]] = []

        full_output = self._run_single_image(
            image,
            prompt,
        )

        candidates.append(
            {
                "region": "full_image",
                "output": full_output,
            }
        )

        for index, tile in enumerate(tiles):

            output = self._run_single_image(
                tile,
                prompt,
            )

            candidates.append(
                {
                    "region": f"tile_{index + 1}",
                    "output": output,
                }
            )

        return {
            "candidates": candidates,
        }

    def postprocess(
        self,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract serial-number candidates from LightOnOCR output.
        """

        extractor = LightOnOCRExtractor()

        regions = []

        for candidate in prediction["candidates"]:

            serial_number = extractor.extract(
                candidate["output"]
            )

            regions.append(
                {
                    "region": candidate["region"],
                    "raw_output": candidate["output"],
                    "serial_number": serial_number,
                }
            )

        return {
            "model": self.model_name,
            "regions": regions,
        }

    def get_model_info(self) -> dict[str, Any]:
        """
        Return backend information.
        """

        return {
            "name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "loaded": self.is_loaded,
            "tile_rows": self.tile_rows,
            "tile_columns": self.tile_columns,
            "tile_overlap": self.tile_overlap,
        }


ModelRegistry.register(
    "lightonocr",
    LightOnOCRBackend,
)