# importing libraries

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from models.backends.lightonocr.lightonocr_backend import (
    LightOnOCRBackend,
)
from utils.ocr_extractor import (
    UniversalOCRExtractor as LightOnOCRExtractor,
)
from utils.ocr_consolidator import (
    UniversalOCRConsolidator as LightOnOCRConsolidator,
)


def print_separator(
    character: str = "=",
    width: int = 70,
) -> None:
    print(character * width)


def print_region_header(
    region_name: str,
) -> None:
    print()
    print(
        f"{'-' * 26} "
        f"{region_name} "
        f"{'-' * 26}"
    )
    print()


def print_raw_outputs(
    candidates: list[dict[str, Any]],
) -> None:
    """
    Display the raw LightOnOCR output exactly as generated.
    """

    print()
    print_separator()
    print("                    LIGHTONOCR RAW OUTPUT")
    print_separator()

    for candidate in candidates:

        region_name = candidate.get(
            "region",
            "unknown",
        )

        raw_output = candidate.get(
            "output",
            "",
        )

        print_region_header(
            region_name
        )

        if raw_output:
            print(
                raw_output.strip()
            )
        else:
            print("NOT_FOUND")

    print()


def print_extraction_results(
    extracted_results: dict[
        str,
        dict[str, Any],
    ],
) -> None:
    """
    Display the deterministic Python extraction
    for every LightOnOCR region.
    """

    print_separator()
    print("                 PYTHON EXTRACTION RESULTS")
    print_separator()

    for region_name, result in extracted_results.items():

        print_region_header(
            region_name
        )

        serial_number = result.get(
            "serial_number",
            "",
        )

        dates = result.get(
            "dates",
            {},
        )

        print(
            "Serial Number : "
            f"{serial_number or 'NOT_FOUND'}"
        )

        print(
            "Dates         : "
            f"{dates or 'NOT_FOUND'}"
        )
        
        imei = result.get(
        "imei",
        "",
        )

        print(
            "IMEI          : "
            f"{imei or 'NOT_FOUND'}"
        )

    print()


def print_final_result(
    final_result: dict[str, Any],
) -> None:
    """
    Display the final consolidated structured result.
    """

    print_separator()
    print("                    FINAL CONSOLIDATED RESULT")
    print_separator()

    serial_number = final_result.get(
        "serial_number",
        "",
    )

    dates = final_result.get(
        "dates",
        {},
    )

    print()
    print(
        "Serial Number : "
        f"{serial_number or 'NOT_FOUND'}"
    )
    
    imei = final_result.get(
        "imei",
        "",
    )

    print(
        "IMEI          : "
        f"{imei or 'NOT_FOUND'}"
    )
    print()
    print("Dates:")

    date_categories = (
        "manufacturing",
        "dated",
        "installation",
        "commissioning",
    )

    for category in date_categories:

        values = dates.get(
            category,
            [],
        )

        if values:

            print(
                f"  {category.capitalize():14}: "
                f"{', '.join(values)}"
            )

        else:

            print(
                f"  {category.capitalize():14}: "
                "NOT_FOUND"
            )

    print()
    print_separator()


def main() -> None:

    if len(sys.argv) < 2:

        print("Usage:")

        print(
            'python -m scripts.test_lightonocr '
            '"data/images/dm_1.png"'
        )

        raise SystemExit(1)

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            f"Image not found: {image_path}"
        )

        raise SystemExit(1)

    print(
        "Loading LightOnOCR-2-1B..."
    )

    backend = LightOnOCRBackend()

    extractor = LightOnOCRExtractor()

    consolidator = LightOnOCRConsolidator()

    try:

        # ----------------------------------------------------------
        # Load model
        # ----------------------------------------------------------

        backend.load(
            Path("model_store/lightonocr")
        )

        print(
            "Model loaded."
        )

        print()
        print(
            "Running 7 image regions..."
        )

        # ----------------------------------------------------------
        # Preprocess image
        # ----------------------------------------------------------
        #
        # This is important because the backend creates:
        #
        # 1 full image
        # 6 overlapping tiles
        #
        # and returns the exact structure expected by predict().
        # ----------------------------------------------------------

        processed_input = backend.preprocess(
            image_path
        )

        # ----------------------------------------------------------
        # Run LightOnOCR
        # ----------------------------------------------------------

        prediction = backend.predict(
            processed_input
        )

        candidates = prediction.get(
            "candidates",
            [],
        )

        # ----------------------------------------------------------
        # Stage 1
        #
        # Display EXACT raw LightOnOCR output.
        # ----------------------------------------------------------

        print_raw_outputs(
            candidates
        )

        # ----------------------------------------------------------
        # Stage 2
        #
        # Run deterministic Python extraction
        # independently on every region.
        # ----------------------------------------------------------

        extracted_results: dict[
            str,
            dict[str, Any],
        ] = {}

        for candidate in candidates:

            region_name = candidate.get(
                "region",
                "unknown",
            )

            raw_output = candidate.get(
                "output",
                "",
            )

            extracted_results[
                region_name
            ] = extractor.extract(
                raw_output
            )

        print_extraction_results(
            extracted_results
        )

        # ----------------------------------------------------------
        # Stage 3
        #
        # Consolidate all region-level results.
        # ----------------------------------------------------------

        final_result = consolidator.consolidate(
            extracted_results
        )

        print_final_result(
            final_result
        )

    finally:

        # ----------------------------------------------------------
        # Always unload the model.
        # ----------------------------------------------------------

        backend.unload()

        print()
        print(
            "Model unloaded."
        )


if __name__ == "__main__":
    main()