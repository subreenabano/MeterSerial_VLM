from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


class UniversalOCRExtractor:
    """
    Model-independent OCR extractor for electricity-meter images.

    Extracts:
        - serial_number
        - imei

    ─────────────────────────────────────────────────────────────
    DESIGN RULES (production-hardened)
    ─────────────────────────────────────────────────────────────

    1. OCR IS THE ONLY SOURCE OF VALUE.
       Filenames are never used to fill in a value. They are only
       used by the caller for audit logging / mismatch detection.

    2. EMPTY IS A VALID ANSWER.
       If the serial or IMEI is not confidently readable (blurry,
       occluded, missing), we return "" with confidence="none".
       We never guess. Wrong values are worse than missing values.

    3. STRICT PATTERN GATE.
         Serial  ->  ^U\\d{6,9}$   OR   ^\\d{7,10}$
         IMEI    ->  ^(86|35|99)\\d{13}$  (15 digits)

    4. MULTI-REGION CONSENSUS.
       Serial must appear in >= 2 OCR regions, OR be a U-prefix
       appearing in >= 1 region (U-prefix is a very strong signal).
       IMEI must appear in >= 1 region.

    5. NEGATIVE CONTEXT REJECTION.
       Lines containing LATITUDE / LONGITUDE / WAN / IMEI /
       NET DATA / dates / readings / contract-no are rejected
       for serial extraction. A "Latitude:15.8091508" line can
       never yield the serial 8091508.

    6. LETTER-BLEED HANDLING.
       OCR often glues a label letter onto the serial:
           "N25367690" -> 25367690
           "No25367690" -> 25367690
       We strip only known bleed letters (N, S, I, O, L, Z, B, G,
       Q, D) and only when U is not the leading letter (U is a
       real Schneider prefix).

    7. BACKWARD-COMPATIBLE API.
       .extract(text)              -> single blob, returns dict
       .extract_from_regions(...)  -> multi-region, adds confidence
    """

    # =========================================================
    # SERIAL LABELS
    # =========================================================

    # Whole-line label patterns (matched with fullmatch after
    # normalization). Used for split-label layouts like:
    #     SL
    #     NO
    #     U5028359
    SERIAL_LABEL_LINE = re.compile(
        r"^\s*(?:"
        r"serial\s*(?:number|no\.?)?|"
        r"sr\.?\s*no\.?|"
        r"sl\.?\s*no\.?|"
        r"sl\.?|"
        r"s\.?\s*no\.?|"
        r"meter\s*(?:number|no\.?|id)|"
        r"mtr\.?\s*(?:no\.?|id)?|"
        r"s\s*/\s*n|"
        r"no\.?"
        r")\s*$",
        re.IGNORECASE,
    )

    # Inline label patterns. Used to split:
    #     SL NO U5028359
    #     Serial Number: ABC12345
    #     Meter No MTR998877
    SERIAL_LABEL_INLINE = re.compile(
        r"\b(?:"
        r"serial\s*(?:number|no\.?)?|"
        r"sr\.?\s*no\.?|"
        r"sl\s*\.?\s*no\.?|"
        r"s\.?\s*no\.?|"
        r"meter\s*(?:number|no\.?|id)|"
        r"mtr\.?\s*(?:no\.?|id)?|"
        r"s\s*/\s*n"
        r")\b",
        re.IGNORECASE,
    )

    IMEI_LABEL_INLINE = re.compile(r"\bimei\b", re.IGNORECASE)

    # Labels whose preceding word disqualifies a standalone "NO"
    NON_SERIAL_NO_CONTEXTS = {
        "IMEI", "CONTRACT", "AWARD", "MODEL", "TYPE",
        "PART", "DATE", "PHONE", "MOBILE",
    }

    # =========================================================
    # NEGATIVE CONTEXT  (line-level rejection)
    # =========================================================

    NEGATIVE_CONTEXT = re.compile(
        r"\b(?:"
        r"lat|latitude|long|longitude|lon|"
        r"wan|imsi|iccid|"
        r"net\s*data|netdata|"
        r"rf|msn|krn|"
        r"firmware|fw|version|"
        r"date|time|timestamp|"
        r"voltage|volt|amp|current|"
        r"kwh|kva|pf|hz|freq|"
        r"gps|temp|temperature|signal|rssi|"
        r"tariff|reading|load|zone|slab|"
        r"https?|www|storage|googleapis|"
        r"contract|award|model|part|"
        r"phone|mobile"
        r")\b",
        re.IGNORECASE,
    )

    DATE_LIKE = re.compile(r"\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}")

    # =========================================================
    # STRICT VALUE PATTERNS
    # =========================================================

    # U + 6-9 digits (Schneider). Optional space after U.
    SERIAL_U_PREFIX = re.compile(r"\bU\s?(\d{6,9})\b")

    # Pure 7-10 digit numeric
    SERIAL_NUMERIC = re.compile(r"\b(\d{7,10})\b")

    # Single leading letter + 7-10 digits (OCR bleed)
    SERIAL_LETTER_BLEED = re.compile(r"\b([A-Z])\s?(\d{7,10})\b")

    # Spaced digits: "2536 7690"
    SERIAL_SPACED = re.compile(r"\b(\d{3,5})[\s\-](\d{3,5})\b")

    # IMEI - 15 digits, must start with 86 / 35 / 99
    IMEI_PATTERN = re.compile(r"\b((?:86|35|99)\d{13})\b")

    # =========================================================
    # LETTERS THAT ARE LIKELY OCR LABEL-BLEED
    # =========================================================

    # Only these are stripped from alphanumerics. U is NOT here —
    # it's the real Schneider prefix.
    BLEED_LETTERS = {"N", "S", "I", "O", "L", "Z", "B", "G", "Q", "D"}

    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(self, min_regions: int = 2) -> None:
        # Minimum number of OCR regions a *numeric* serial must
        # appear in to be accepted. U-prefixed serials bypass this
        # because U-prefix is already a very strong signal.
        self.min_regions = min_regions

    # =========================================================
    # PUBLIC API #1 - SINGLE BLOB (backward compatible)
    # =========================================================

    def extract(self, text: str | None) -> dict[str, Any]:
        """
        Extract serial + IMEI from a single OCR text blob.

        Returns:
            {"serial_number": str, "imei": str}

        Empty string means "not found / not confident".
        """
        if not text:
            return {"serial_number": "", "imei": ""}

        lines = self._prepare_lines(text)
        return {
            "serial_number": self._extract_serial(lines),
            "imei": self._extract_imei(lines),
        }

    # =========================================================
    # PUBLIC API #2 - MULTI-REGION (production entry point)
    # =========================================================

    def extract_from_regions(
        self, region_outputs: dict[str, str]
    ) -> dict[str, Any]:
        """
        Extract using cross-region consensus.

        region_outputs: mapping region_name -> OCR text, e.g.
            {
                "full_image": "...",
                "tile_1": "...",
                "tile_2": "...",
                ...
            }

        Returns:
            {
                "serial_number": str,
                "imei": str,
                "serial_confidence": "high" | "medium" | "low" | "none",
                "serial_evidence": {
                    "regions": [str, ...],
                    "count": int,
                    "kind": "u_prefix" | "numeric" | "letter_bleed" | "spaced"
                }
            }

        Serial is empty when evidence is weak (single numeric region).
        IMEI is empty when not found.
        """
        serial_scores: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"regions": set(), "kind": "", "value": ""}
        )
        imei_scores: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"regions": set(), "value": ""}
        )

        for region, text in region_outputs.items():
            if not text:
                continue

            lines = self._prepare_lines(text)

            for value, kind in self._serial_candidates(lines):
                key = value.upper()
                entry = serial_scores[key]
                entry["regions"].add(region)
                entry["kind"] = kind
                entry["value"] = value

            for value in self._imei_candidates(lines):
                entry = imei_scores[value]
                entry["regions"].add(region)
                entry["value"] = value

        # ---- pick best serial ----
        best_serial = ""
        serial_conf = "none"
        serial_ev: dict[str, Any] = {}

        u_candidates = [
            (k, v) for k, v in serial_scores.items() if k.startswith("U")
        ]
        num_candidates = [
            (k, v)
            for k, v in serial_scores.items()
            if k.isdigit() and len(v["regions"]) >= self.min_regions
        ]

        chosen: tuple[str, dict[str, Any]] | None = None

        if u_candidates:
            # Prefer U-prefix; pick one with most regions
            chosen = max(u_candidates, key=lambda kv: len(kv[1]["regions"]))
        elif num_candidates:
            chosen = max(num_candidates, key=lambda kv: len(kv[1]["regions"]))

        if chosen:
            n = len(chosen[1]["regions"])
            best_serial = chosen[1]["value"]
            if n >= 4:
                serial_conf = "high"
            elif n >= 2:
                serial_conf = "medium"
            else:
                serial_conf = "low"
            serial_ev = {
                "regions": sorted(chosen[1]["regions"]),
                "count": n,
                "kind": chosen[1]["kind"],
            }

        # ---- pick best IMEI ----
        best_imei = ""
        if imei_scores:
            chosen_imei = max(
                imei_scores.items(),
                key=lambda kv: len(kv[1]["regions"]),
            )
            # Require at least 1 region for IMEI (very distinctive pattern)
            if len(chosen_imei[1]["regions"]) >= 1:
                best_imei = chosen_imei[1]["value"]

        return {
            "serial_number": best_serial,
            "imei": best_imei,
            "serial_confidence": serial_conf,
            "serial_evidence": serial_ev,
        }

    # =========================================================
    # LINE PREPARATION
    # =========================================================

    def _prepare_lines(self, text: str) -> list[str]:
        """Normalize line endings, strip markdown, collapse whitespace."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        out: list[str] = []
        for line in text.split("\n"):
            line = re.sub(r"[*_`#]+", " ", line)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                out.append(line)
        return out

    # =========================================================
    # SERIAL EXTRACTION - SINGLE BLOB
    # =========================================================

    def _extract_serial(self, lines: list[str]) -> str:
        candidates = self._serial_candidates(lines)
        if not candidates:
            return ""

        # Prefer U-prefix (Schneider)
        for value, kind in candidates:
            if kind == "u_prefix":
                return value

        # Then letter-bleed, spaced, numeric
        for value, kind in candidates:
            if kind in {"letter_bleed", "spaced", "numeric"}:
                return value

        return ""

    # =========================================================
    # SERIAL EXTRACTION - CANDIDATE GENERATION
    # =========================================================

    def _serial_candidates(
        self, lines: list[str]
    ) -> list[tuple[str, str]]:
        """
        Return list of (value, kind) pairs.

        kind in {"u_prefix", "numeric", "letter_bleed", "spaced"}.
        Value is normalized (U#### for Schneider, digits for others).
        """
        candidates: list[tuple[str, str]] = []

        # ---------------------------------------------
        # PASS 1 - INLINE LABELS  (SL NO U5028359)
        # ---------------------------------------------
        for line in lines:
            if self.IMEI_LABEL_INLINE.search(line):
                continue  # Never let IMEI lines feed serial
            if self._is_negative_line(line):
                continue

            for m in self.SERIAL_LABEL_INLINE.finditer(line):
                # Reject "CONTRACT NO", "AWARD NO", etc.
                prefix_words = line[: m.start()].strip().upper().split()
                if prefix_words:
                    last = prefix_words[-1].rstrip(".:")
                    if last in self.NON_SERIAL_NO_CONTEXTS:
                        continue

                rest = line[m.end():].lstrip(":.-# ").strip()
                value, kind = self._parse_serial_value(rest)
                if value:
                    candidates.append((value, kind))

        # ---------------------------------------------
        # PASS 2 - SPLIT LABELS  (SL / NO / U5028359)
        # ---------------------------------------------
        for i in range(len(lines)):
            # Bare label line: "SL", "NO.", "SERIAL"
            if self.SERIAL_LABEL_LINE.match(lines[i]):
                # Reject if previous line is IMEI/CONTRACT/etc.
                if i > 0:
                    prev_words = lines[i - 1].strip().upper().split()
                    if prev_words:
                        last = prev_words[-1].rstrip(".:")
                        if last in self.NON_SERIAL_NO_CONTEXTS:
                            continue
                    if re.search(
                        r"\bimei\b", lines[i - 1], re.IGNORECASE
                    ):
                        continue

                if i + 1 < len(lines):
                    value, kind = self._parse_serial_value(lines[i + 1])
                    if value:
                        candidates.append((value, kind))

            # Combined two-line label: "SL" + "NO" then value
            if i + 2 < len(lines):
                combined = f"{lines[i]} {lines[i + 1]}".upper().strip()
                if combined in {
                    "SL NO", "SL NO.", "SL. NO", "SL. NO.",
                    "S NO", "S NO.", "S. NO", "S. NO.",
                    "SERIAL NUMBER", "SERIAL NO", "SERIAL NO.",
                    "METER NUMBER", "METER NO", "METER NO.",
                }:
                    value, kind = self._parse_serial_value(lines[i + 2])
                    if value:
                        candidates.append((value, kind))

        # ---------------------------------------------
        # PASS 3 - BARE U-PREFIX ANYWHERE
        # U-prefix is such a strong signal that even a
        # bare occurrence (no label nearby) counts.
        # ---------------------------------------------
        for line in lines:
            if self.IMEI_LABEL_INLINE.search(line):
                continue
            if self._is_negative_line(line):
                continue
            for m in self.SERIAL_U_PREFIX.finditer(line):
                candidates.append((f"U{m.group(1)}", "u_prefix"))

        return candidates

    # =========================================================
    # SERIAL VALUE PARSER
    # =========================================================

    def _parse_serial_value(self, text: str) -> tuple[str, str]:
        """
        Extract ONE serial-shaped value from a text fragment.

        Priority:
            1. U-prefix        U5773434
            2. Letter-bleed    N25367690 -> 25367690
            3. Spaced digits   2536 7690  -> 25367690
            4. Pure numeric    25367690

        Returns ("", "") if nothing valid found.
        """
        if not text:
            return "", ""

        text = re.sub(r"^[\s:;,\-./]+", "", text.strip())
        if self._is_negative_line(text):
            return "", ""

        # ---- Priority 1: U-prefix ----
        m = self.SERIAL_U_PREFIX.search(text)
        if m:
            return f"U{m.group(1)}", "u_prefix"

        # ---- Priority 2: letter-bleed ----
        for m in self.SERIAL_LETTER_BLEED.finditer(text):
            letter = m.group(1).upper()
            digits = m.group(2)
            if letter == "U":
                continue  # U handled above
            if letter not in self.BLEED_LETTERS:
                continue
            # Ensure it's a whole token (not "MTR1234567")
            start, _ = m.span()
            if start == 0 or not text[start - 1].isalnum():
                return digits, "letter_bleed"

        # ---- Priority 3: spaced digits ----
        for m in self.SERIAL_SPACED.finditer(text):
            a, b = m.group(1), m.group(2)
            combined = a + b
            if 7 <= len(combined) <= 10:
                return combined, "spaced"

        # ---- Priority 4: pure numeric ----
        m = self.SERIAL_NUMERIC.search(text)
        if m:
            return m.group(1), "numeric"

        return "", ""

    # =========================================================
    # IMEI EXTRACTION
    # =========================================================

    def _extract_imei(self, lines: list[str]) -> str:
        cands = self._imei_candidates(lines)
        if not cands:
            return ""
        # Prefer 15-digit starting with 86
        for c in cands:
            if c.startswith("86") and len(c) == 15:
                return c
        return cands[0]

    def _imei_candidates(self, lines: list[str]) -> list[str]:
        out: list[str] = []

        # ---- Inline: "IMEI NO: 861729077002436" ----
        for line in lines:
            if self.IMEI_LABEL_INLINE.search(line):
                for m in self.IMEI_PATTERN.finditer(line):
                    out.append(m.group(1))

        # ---- Split: "IMEI" / "861729077002436" ----
        for i, line in enumerate(lines):
            if not re.search(r"^\s*imei\b", line, re.IGNORECASE):
                continue
            for offset in (1, 2):
                j = i + offset
                if j >= len(lines):
                    break
                m = self.IMEI_PATTERN.search(lines[j])
                if m:
                    out.append(m.group(1))
                    break

        # ---- Fallback: bare 15-digit starting with 86/35/99 ----
        for line in lines:
            if self._is_negative_line(line):
                continue
            for m in self.IMEI_PATTERN.finditer(line):
                out.append(m.group(1))

        return out

    # =========================================================
    # NEGATIVE CONTEXT
    # =========================================================

    def _is_negative_line(self, line: str) -> bool:
        if self.NEGATIVE_CONTEXT.search(line):
            return True
        if self.DATE_LIKE.search(line):
            return True
        return False

    # =========================================================
    # FILENAME HINT - FOR AUDIT LOGGING ONLY
    # =========================================================

    @staticmethod
    def extract_filename_hint(filename: str) -> str:
        """
        Extract a possible serial from the filename for AUDIT
        LOGGING ONLY. Never use the result as the serial value
        - the filename is not a source of truth.
        """
        if not filename:
            return ""
        m = re.search(r"_([UN]\d{6,10})_", filename, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.search(r"_(\d{7,10})_newMtr", filename, re.IGNORECASE)
        if m:
            return m.group(1)
        return ""