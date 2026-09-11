import pandas as pd
from pathlib import Path

PRED_CSV = "predictions_test.csv"
REPORT_TXT = "evaluation_report.txt"

def normalize(val):
    if pd.isna(val):
        return ""
    v = str(val).strip().upper()
    # Strip trailing .0 from any accidental float conversion
    if v.endswith(".0"):
        v = v[:-2]
    return v

def is_found(val):
    v = normalize(val)
    return v not in ("", "NOT_FOUND", "NONE", "NAN")

def char_accuracy(truth, pred):
    if not truth or not pred:
        return 0.0
    match = sum(1 for a, b in zip(truth, pred) if a == b)
    return match / max(len(truth), len(pred))

df = pd.read_csv(PRED_CSV, dtype={"true_serial": str, "pred_serial": str, "true_imei": str, "pred_imei": str})

def evaluate_field(field_name):
    true_col = f"true_{field_name}"
    pred_col = f"pred_{field_name}"

    TP = TN = FP = FN = 0
    char_accs = []

    for _, row in df.iterrows():
        t = normalize(row[true_col])
        p = normalize(row[pred_col])
        t_found = is_found(t)
        p_found = is_found(p)

        if t_found and p_found and t == p:
            TP += 1
            char_accs.append(1.0)
        elif t_found and p_found and t != p:
            FP += 1
            FN += 1
            char_accs.append(char_accuracy(t, p))
        elif t_found and not p_found:
            FN += 1
            char_accs.append(0.0)
        elif not t_found and p_found:
            FP += 1
        else:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (TP + TN) / len(df) if len(df) else 0.0
    avg_char  = sum(char_accs) / len(char_accs) if char_accs else 0.0

    return {
        "field": field_name,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "char_accuracy": avg_char,
    }

results = [evaluate_field("serial"), evaluate_field("imei")]

lines = []
lines.append("=" * 70)
lines.append("           PADDLEOCR EVALUATION REPORT")
lines.append("=" * 70)
lines.append(f"Total images evaluated: {len(df)}")
lines.append("")

for r in results:
    lines.append("-" * 70)
    lines.append(f"FIELD: {r['field'].upper()}")
    lines.append("-" * 70)
    lines.append("  Confusion Matrix:")
    lines.append(f"     TP: {r['TP']}    TN: {r['TN']}")
    lines.append(f"     FP: {r['FP']}    FN: {r['FN']}")
    lines.append("")
    lines.append("  Metrics:")
    lines.append(f"     Accuracy         : {r['accuracy']*100:.2f}%")
    lines.append(f"     Precision        : {r['precision']*100:.2f}%")
    lines.append(f"     Recall           : {r['recall']*100:.2f}%")
    lines.append(f"     F1-Score         : {r['f1']*100:.2f}%")
    lines.append(f"     Avg Char Accuracy: {r['char_accuracy']*100:.2f}%")
    lines.append("")

lines.append("=" * 70)
lines.append("              PER-IMAGE BREAKDOWN")
lines.append("=" * 70)
for _, row in df.iterrows():
    t_s = normalize(row["true_serial"]); p_s = normalize(row["pred_serial"])
    t_i = normalize(row["true_imei"]);   p_i = normalize(row["pred_imei"])
    ok_s = "✅" if t_s == p_s else "❌"
    ok_i = "✅" if t_i == p_i else "❌"
    lines.append(f"\n{row['filename']}")
    lines.append(f"   Serial: True={t_s:<20} Pred={p_s:<20} {ok_s}")
    lines.append(f"   IMEI  : True={t_i:<20} Pred={p_i:<20} {ok_i}")

report = "\n".join(lines)
Path(REPORT_TXT).write_text(report, encoding="utf-8")
print(report)
print(f"\n📄 Full report saved to {REPORT_TXT}")