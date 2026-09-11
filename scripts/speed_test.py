import time
from pathlib import Path
from PIL import Image
from models.backends.paddleocr.paddleocr_backend import PaddleOCRBackend

IMG = next(Path("data/test_10").glob("*.png"))
print(f"Testing: {IMG.name}")

img = Image.open(IMG)
print(f"Original: {img.size[0]} x {img.size[1]}")

t0 = time.time()
backend = PaddleOCRBackend()
backend.load()
print(f"Model load: {time.time() - t0:.1f}s")

processed = backend.preprocess(IMG)

t2 = time.time()
backend._run_ocr(processed['image'])
full_time = time.time() - t2
print(f"Full-image OCR: {full_time:.1f}s")

t3 = time.time()
backend._run_ocr(processed['tiles'][0])
tile_time = time.time() - t3
print(f"Single-tile OCR: {tile_time:.1f}s")

estimate = full_time + 6 * tile_time
print(f"\n✅ Estimated per image: ~{estimate:.1f}s")
print(f"✅ Estimated for 1,998 images: ~{estimate * 1998 / 3600:.1f} hours")