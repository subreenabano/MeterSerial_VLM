        # ---- PASS 5: BARE 8-DIGIT NUMERIC NEAR A SERIAL LABEL ----
        for i, line in enumerate(lines):
            stripped = line.strip()

            if not re.fullmatch(r"\d{8}", stripped):
                continue

            if self.IMEI_LABEL_INLINE.search(line):
                continue
            if self._is_negative_line(line):
                continue

            window_start = max(0, i - 2)
            window_end = min(len(lines), i + 3)
            has_label_nearby = False

            for j in range(window_start, window_end):
                if j == i:
                    continue
                if self.SERIAL_LABEL_LINE.match(lines[j]):
                    has_label_nearby = True
                    break

            if has_label_nearby:
                candidates.append((stripped, "numeric"))