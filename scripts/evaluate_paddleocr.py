import pandas as pd
import time
from pathlib import Path
from models.backends.paddleocr.paddleocr_backend import PaddleOCRBackend
from utils.ocr_extractor import UniversalOCRExtractor
from utils.ocr_consolidator import UniversalOCRConsolidator

# --- Configuration ---
IMAGE_DIR = Path("data/test_10")
GROUND_TRUTH = Path("data/test_10/ground_truth.csv")
PRED_CSV = "predictions_test.csv"


def run_predictions():
    if not GROUND_TRUTH.exists():
        print(f"❌ Ground truth file not found: {GROUND_TRUTH}")
        return

    gt_df = pd.read_csv(GROUND_TRUTH)
    print(f"Found {len(gt_df)} ground truth entries.\n")

    print("Loading PaddleOCR...")
    backend = PaddleOCRBackend()
    backend.load()
    extractor = UniversalOCRExtractor()
    consolidator = UniversalOCRConsolidator()

    results = []
    start = time.time()

    for i, row in gt_df.iterrows():
        img_path = IMAGE_DIR / row["filename"]
        if not img_path.exists():
            print(f"⚠️ Missing image: {img_path}")
            continue

        print(f"[{i+1}/{len(gt_df)}] {img_path.name}")
        try:
            processed = backend.preprocess(img_path)
            raw = backend.predict(processed)
            final = backend.postprocess(raw)

            # -------- PER-REGION EXTRACTION --------
            region_results = {}
            for r in final["regions"]:
                region_name = r["region"]
                raw_output = r["raw_output"]
                region_results[region_name] = extractor.extract(raw_output)

            # -------- CROSS-REGION CONSOLIDATION --------
            final_result = consolidator.consolidate(region_results)
            pred_serial = final_result.get("serial_number", "")
            pred_imei = final_result.get("imei", "")

            results.append({
                "filename": img_path.name,
                "true_serial": str(row["true_serial"]) if pd.notna(row["true_serial"]) else "",
                "pred_serial": pred_serial,
                "true_imei": str(row["true_imei"]) if pd.notna(row["true_imei"]) else "",
                "pred_imei": pred_imei,
            })
            print(f"   → Serial: {pred_serial} | IMEI: {pred_imei}\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    pred_df = pd.DataFrame(results)
    pred_df.to_csv(PRED_CSV, index=False)
    elapsed = time.time() - start
    print(f"✅ Predictions saved to {PRED_CSV}")
    print(f"⏱️ Total time: {elapsed/60:.2f} min ({elapsed/max(len(results),1):.1f}s per image)")


if __name__ == "__main__":
    run_predictions()