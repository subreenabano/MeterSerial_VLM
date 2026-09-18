from __future__ import annotations

import re
from typing import Any


class UniversalOCRExtractor:
    """
    Model-independent OCR extractor for electricity-meter OCR.

    Extracts:
        - serial_number
        - imei

    Designed to work with OCR output from:
        - PaddleOCR
        - LightOnOCR
        - future OCR/VLM models

    Important design principle:

        We only accept a serial number when there is
        sufficient textual evidence that the value is
        actually associated with a serial-number label.

    Supported examples:

        SL NO U5028359
        SL. NO. U5028359

        SL
        NO
        U5028359

        SL.
        U5028359

        NO. U5023881

        NO
        U5023881

        Serial Number: U5028359

        Meter No MTR998877

        S/N
        U5028359

    IMEI examples:

        IMEI NO: 884524071203313

        IMEI
        NO
        884524071203313
    """

    # =========================================================
    # Serial-number labels
    # =========================================================

    SERIAL_LABEL_PATTERNS = [

        # Serial Number
        re.compile(
            r"^\s*serial\s+number\s*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*serial\s+no\.?\s*$",
            re.IGNORECASE,
        ),

        # Meter Number
        re.compile(
            r"^\s*meter\s+number\s*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*meter\s+no\.?\s*$",
            re.IGNORECASE,
        ),

        # Meter ID
        re.compile(
            r"^\s*meter\s+id\s*$",
            re.IGNORECASE,
        ),

        # S/N
        re.compile(
            r"^\s*s\s*/\s*n\s*$",
            re.IGNORECASE,
        ),

        # SL NO
        re.compile(
            r"^\s*sl\s+no\.?\s*$",
            re.IGNORECASE,
        ),

        # SL.
        re.compile(
            r"^\s*sl\.?\s*$",
            re.IGNORECASE,
        ),

        # S. NO.
        re.compile(
            r"^\s*s\.?\s+no\.?\s*$",
            re.IGNORECASE,
        ),

        # Standalone NO / NO.
        #
        # This is intentionally anchored.
        #
        # It will NOT match:
        #   NO DATA
        #   IMEI NO
        #   CONTRACT NO
        #
        re.compile(
            r"^\s*no\.?\s*$",
            re.IGNORECASE,
        ),
    ]

    # =========================================================
    # IMEI labels
    # =========================================================

    IMEI_LABEL_PATTERNS = [

        re.compile(
            r"^\s*imei\s*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*imei\s+no\.?\s*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*imei\s+number\s*$",
            re.IGNORECASE,
        ),
    ]

    # =========================================================
    # Identifier patterns
    # =========================================================

    # Typical meter serial numbers.
    #
    # Examples:
    #   U5028359
    #   MTR998877
    #   ABC12345
    #   S3DKLG0S9601
    #
    SERIAL_VALUE_PATTERN = re.compile(
        r"\b[A-Z0-9][A-Z0-9\-\/]{4,24}\b",
        re.IGNORECASE,
    )

    # Standard IMEI is 15 digits.
    #
    # We allow 14-17 digits because OCR can occasionally
    # lose or duplicate a digit.
    IMEI_VALUE_PATTERN = re.compile(
        r"\b\d{14,17}\b"
    )

    # =========================================================
    # Words that are clearly not serial numbers
    # =========================================================

    SERIAL_EXCLUSIONS = {
        "PHASE",
        "WIRE",
        "METER",
        "SMART",
        "STATIC",
        "WATTHOUR",
        "MANUFACTURED",
        "MANUFACTURER",
        "NUMBER",
        "NO",
        "SL",
        "SERIAL",
        "IMEI",
        "DATE",
        "DIAL",
        "NET",
        "DATA",
        "AC",
        "KW",
        "KWH",
        "KVA",
        "VOLT",
        "VOLTAGE",
        "CURRENT",
        "FREQUENCY",
        "LATITUDE",
        "LONGITUDE",
        "WAN",
        "AURORA",
        "SCHNEIDER",
        "ELECTRIC",
        "INDIA",
        "PRIVATE",
        "LIMITED",
        "MYSURU",
        "CONTRACT",
        "AWARD",
        "TYPE",
        "DLMS",
        "CAT",
        "WARRANTY",
        "PERIOD",
        "PROPERTY",
        "BESCOM",
        "SCROLL",
        "BATT",
    }

    # Words which indicate that a standalone NO belongs
    # to something other than the meter serial number.
    NON_SERIAL_NO_CONTEXTS = {
        "IMEI",
        "CONTRACT",
        "AWARD",
        "MODEL",
        "TYPE",
        "PART",
        "DATE",
        "PHONE",
        "MOBILE",
    }

    def __init__(
        self,
        context_window: int = 1,
    ) -> None:

        self.context_window = context_window

    # =========================================================
    # Public API
    # =========================================================

    def extract(
        self,
        text: str | None,
    ) -> dict[str, Any]:

        if not text:

            return {
                "serial_number": "",
                "imei": "",
            }

        lines = self._prepare_lines(
            text
        )

        serial_candidates = (
            self._extract_serial_candidates(
                lines
            )
        )

        imei_candidates = (
            self._extract_imei_candidates(
                lines
            )
        )

        return {
            "serial_number": (
                serial_candidates[0]
                if serial_candidates
                else ""
            ),
            "imei": (
                imei_candidates[0]
                if imei_candidates
                else ""
            ),
        }

    # =========================================================
    # Prepare OCR text
    # =========================================================

    def _prepare_lines(
        self,
        text: str,
    ) -> list[str]:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        raw_lines = text.split(
            "\n"
        )

        lines: list[str] = []

        for line in raw_lines:

            line = line.strip()

            if not line:
                continue

            # Remove common Markdown formatting
            # generated by VLM OCR.
            line = re.sub(
                r"[*_`#]+",
                " ",
                line,
            )

            # Normalize whitespace.
            line = re.sub(
                r"\s+",
                " ",
                line,
            ).strip()

            if line:
                lines.append(line)

        return lines

    # =========================================================
    # Serial extraction
    # =========================================================

    def _extract_serial_candidates(
        self,
        lines: list[str],
    ) -> list[str]:

        candidates: list[str] = []

        # -----------------------------------------------------
        # PASS 1
        #
        # Same-line labels.
        #
        # Examples:
        #
        # SL NO U5028359
        # Serial Number: ABC12345
        # Meter No MTR998877
        # -----------------------------------------------------

        for index, line in enumerate(lines):

            candidate = (
                self._extract_serial_from_same_line(
                    line
                )
            )

            if candidate:

                candidates.append(
                    candidate
                )

        # -----------------------------------------------------
        # PASS 2
        #
        # Split labels.
        #
        # SL
        # NO
        # U5028359
        #
        # Serial
        # Number
        # ABC12345
        #
        # SL.
        # U5023881
        # -----------------------------------------------------

        for index in range(
            len(lines)
        ):

            # -----------------------------------------------
            # Two-line label:
            #
            # SL
            # NO
            # -----------------------------------------------

            if (
                index + 1
                < len(lines)
            ):

                combined_label = (
                    f"{lines[index]} "
                    f"{lines[index + 1]}"
                )

                if self._is_combined_serial_label(
                    combined_label
                ):

                    # The value MUST be immediately
                    # after the label.
                    value_index = (
                        index + 2
                    )

                    if (
                        value_index
                        < len(lines)
                    ):

                        candidate = (
                            self._find_serial_value(
                                lines[
                                    value_index
                                ]
                            )
                        )

                        if candidate:

                            candidates.append(
                                candidate
                            )

            # -----------------------------------------------
            # Single-line label:
            #
            # SL.
            # U5023881
            #
            # NO.
            # U5023881
            #
            # S/N
            # U5023881
            # -----------------------------------------------

            if self._is_serial_label(
                lines[index]
            ):

                # IMEI NO must never become serial.
                if self._is_imei_related_label(
                    lines,
                    index,
                ):
                    continue

                value_index = (
                    index + 1
                )

                if (
                    value_index
                    >= len(lines)
                ):
                    continue

                candidate = (
                    self._find_serial_value(
                        lines[
                            value_index
                        ]
                    )
                )

                if candidate:

                    candidates.append(
                        candidate
                    )

        return self._unique(
            candidates
        )

    # =========================================================
    # Same-line serial extraction
    # =========================================================

    def _extract_serial_from_same_line(
        self,
        line: str,
    ) -> str:

        # -----------------------------------------------------
        # IMEI line must be ignored by serial extraction.
        # -----------------------------------------------------

        if re.search(
            r"\bimei\b",
            line,
            re.IGNORECASE,
        ):

            return ""

        # -----------------------------------------------------
        # Check each serial label.
        # -----------------------------------------------------

        label_patterns = [

            re.compile(
                r"\bserial\s+number\b",
                re.IGNORECASE,
            ),

            re.compile(
                r"\bserial\s+no\.?\b",
                re.IGNORECASE,
            ),

            re.compile(
                r"\bmeter\s+number\b",
                re.IGNORECASE,
            ),

            re.compile(
                r"\bmeter\s+no\.?\b",
                re.IGNORECASE,
            ),

            re.compile(
                r"\bmeter\s+id\b",
                re.IGNORECASE,
            ),

            re.compile(
                r"\bs\s*/\s*n\b",
                re.IGNORECASE,
            ),

            re.compile(
                r"\bsl\s*\.?\s*no\.?\b",
                re.IGNORECASE,
            ),

            re.compile(
                r"\bs\.?\s+no\.?\b",
                re.IGNORECASE,
            ),
        ]

        for pattern in label_patterns:

            match = pattern.search(
                line
            )

            if not match:
                continue

            remainder = line[
                match.end():
            ].strip()

            remainder = remainder.lstrip(
                ":.-#"
            ).strip()

            candidate = (
                self._find_serial_value(
                    remainder
                )
            )

            if candidate:

                return candidate

        return ""

    # =========================================================
    # Label checks
    # =========================================================

    def _is_serial_label(
        self,
        text: str,
    ) -> bool:

        normalized = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        for pattern in (
            self.SERIAL_LABEL_PATTERNS
        ):

            if pattern.fullmatch(
                normalized
            ):

                return True

        return False

    def _is_combined_serial_label(
        self,
        text: str,
    ) -> bool:

        normalized = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        normalized = normalized.upper()

        valid = {
            "SL NO",
            "SL NO.",
            "SL. NO",
            "SL. NO.",
            "S NO",
            "S NO.",
            "S. NO",
            "S. NO.",
            "SERIAL NUMBER",
            "SERIAL NO",
            "SERIAL NO.",
            "METER NUMBER",
            "METER NO",
            "METER NO.",
        }

        return normalized in valid

    # =========================================================
    # Determine whether NO belongs to IMEI
    # =========================================================

    def _is_imei_related_label(
        self,
        lines: list[str],
        index: int,
    ) -> bool:

        current = lines[
            index
        ].strip()

        if re.search(
            r"\bimei\b",
            current,
            re.IGNORECASE,
        ):

            return True

        # Check immediately preceding line.
        if index > 0:

            previous = lines[
                index - 1
            ].strip()

            if re.search(
                r"\bimei\b",
                previous,
                re.IGNORECASE,
            ):

                return True

        return False

    # =========================================================
    # Find serial value
    # =========================================================

    def _find_serial_value(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        # Never extract IMEI as serial.
        if re.search(
            r"\bimei\b",
            text,
            re.IGNORECASE,
        ):

            return ""

        cleaned = text.strip()

        # Remove common separators.
        cleaned = re.sub(
            r"^[\s:;,\-./]+",
            "",
            cleaned,
        )

        matches = (
            self.SERIAL_VALUE_PATTERN.findall(
                cleaned
            )
        )

        for value in matches:

            normalized = (
                self._normalize_identifier(
                    value
                )
            )

            if not normalized:
                continue

            upper_value = (
                normalized.upper()
            )

            # ---------------------------------------------
            # Reject known non-serial words.
            # ---------------------------------------------

            if (
                upper_value
                in self.SERIAL_EXCLUSIONS
            ):

                continue

            # ---------------------------------------------
            # Reject pure short alphabetic words.
            #
            # This prevents:
            #
            # NO → no
            # AURO → etc.
            # ---------------------------------------------

            if (
                normalized.isalpha()
                and len(normalized) < 6
            ):

                continue

            # ---------------------------------------------
            # Reject pure numeric values here.
            #
            # Meter serials in this project can be
            # alphanumeric, but numeric-only values
            # are much more likely to be readings,
            # dates, contract numbers, etc.
            #
            # We allow numeric serials only when they
            # are long enough to plausibly be a serial.
            # ---------------------------------------------

            if normalized.isdigit():

                if len(normalized) < 6:
                    continue

            return normalized

        return ""

    # =========================================================
    # IMEI extraction
    # =========================================================

    def _extract_imei_candidates(
        self,
        lines: list[str],
    ) -> list[str]:

        candidates: list[str] = []

        # -----------------------------------------------------
        # PASS 1
        #
        # IMEI on same line.
        #
        # IMEI NO: 884524071203313
        # -----------------------------------------------------

        for line in lines:

            if not re.search(
                r"\bimei\b",
                line,
                re.IGNORECASE,
            ):

                continue

            candidate = (
                self._find_imei_value(
                    line
                )
            )

            if candidate:

                candidates.append(
                    candidate
                )

        # -----------------------------------------------------
        # PASS 2
        #
        # Split IMEI.
        #
        # IMEI
        # NO
        # 884524071203313
        #
        # IMEI NO
        # 884524071203313
        # -----------------------------------------------------

        for index, line in enumerate(
            lines
        ):

            if not re.search(
                r"\bimei\b",
                line,
                re.IGNORECASE,
            ):

                continue

            # Search ONLY the next two lines.
            #
            # We do not search far away because
            # unrelated numbers may appear later.
            for offset in (
                1,
                2,
            ):

                value_index = (
                    index + offset
                )

                if (
                    value_index
                    >= len(lines)
                ):

                    break

                candidate = (
                    self._find_imei_value(
                        lines[
                            value_index
                        ]
                    )
                )

                if candidate:

                    candidates.append(
                        candidate
                    )

                    break

        return self._unique(
            candidates
        )

    # =========================================================
    # Find IMEI value
    # =========================================================

    def _find_imei_value(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        matches = (
            self.IMEI_VALUE_PATTERN.findall(
                text
            )
        )

        if not matches:
            return ""

        # Prefer standard 15-digit IMEI.
        fifteen_digit = [
            value
            for value in matches
            if len(value) == 15
        ]

        if fifteen_digit:

            return fifteen_digit[0]

        return matches[0]

    # =========================================================
    # Utility methods
    # =========================================================

    @staticmethod
    def _normalize_identifier(
        value: str,
    ) -> str:

        value = value.strip()

        value = re.sub(
            r"[.,:;]+$",
            "",
            value,
        )

        return value

    @staticmethod
    def _unique(
        values: list[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for value in values:

            key = value.upper()

            if key in seen:
                continue

            seen.add(key)

            result.append(
                value
            )

        return result