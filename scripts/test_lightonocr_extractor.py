# importing libraries

from utils.ocr_extractor import (
    UniversalOCRExtractor as LightOnOCRExtractor,
)


def main() -> None:

    extractor = LightOnOCRExtractor()

    test_cases = [
        {
            "text": "SL NO U5028359",
            "expected": {
                "serial_number": "U5028359",
                "imei": "",
                "dates": {},
            },
        },
        {
            "text": (
                "S. NO. 20258635\n"
                "MFG 03/2025"
            ),
            "expected": {
                "serial_number": "20258635",
                "imei": "",
                "dates": {
                    "manufacturing": [
                        "03/2025"
                    ]
                },
            },
        },
        {
            "text": (
                "Serial Number: ABC12345\n"
                "Dated: 23.12.2024"
            ),
            "expected": {
                "serial_number": "ABC12345",
                "imei": "",
                "dates": {
                    "dated": [
                        "23.12.2024"
                    ]
                },
            },
        },
        {
            "text": (
                "Meter No MTR998877\n"
                "Installation Date: 15/10/2025"
            ),
            "expected": {
                "serial_number": "MTR998877",
                "imei": "",
                "dates": {
                    "installation": [
                        "15/10/2025"
                    ]
                },
            },
        },
        {
            "text": (
                "SL NO U5028359\n"
                "IMEI NO 86073807944914\n"
                "MFG 02/2025"
            ),
            "expected": {
                "serial_number": "U5028359",
                "imei": "86073807944914",
                "dates": {
                    "manufacturing": [
                        "02/2025"
                    ]
                },
            },
        },
        {
            "text": (
                "Serial Number: ABC12345\n"
                "IMEI: 123456789012345\n"
                "Dated: 23.12.2024\n"
                "Installation Date: 15/10/2025"
            ),
            "expected": {
                "serial_number": "ABC12345",
                "imei": "123456789012345",
                "dates": {
                    "dated": [
                        "23.12.2024"
                    ],
                    "installation": [
                        "15/10/2025"
                    ],
                },
            },
        },
    ]

    passed = 0

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):

        result = extractor.extract(
            test_case["text"]
        )

        expected = test_case["expected"]

        print("=" * 70)
        print(
            f"TEST {index}"
        )

        print(
            "OCR INPUT"
        )

        print(
            test_case["text"]
        )

        print()
        print(
            "RESULT"
        )

        print(result)

        print()
        print(
            "EXPECTED"
        )

        print(expected)

        if result == expected:

            print()
            print("PASS")

            passed += 1

        else:

            print()
            print("FAIL")

    print()
    print("=" * 70)

    print(
        f"{passed}/{len(test_cases)} tests passed"
    )

    if passed != len(test_cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()