import os
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

# Configuration
URL_FILE = "urls.txt"
OUTPUT_DIR = Path("data/images")
MAX_WORKERS = 15  # Downloads 15 images at the exact same time

def download_image(url):
    try:
        url = url.strip()
        # Skip the Excel header row if it got copied by mistake
        if not url.startswith("http"):
            return False
            
        # Extract filename from the URL
        filename = url.split("/")[-1]
        if not filename.endswith(".png"):
            filename += ".png"
            
        filepath = OUTPUT_DIR / filename
        
        # Skip if we already downloaded it (saves time if you have to restart)
        if filepath.exists():
            return True

        # Download the image
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            return True
        return False
    except Exception:
        return False

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not Path(URL_FILE).exists():
        print(f"❌ Error: Could not find {URL_FILE}. Please make sure you created it in the root folder.")
        return

    with open(URL_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to download. Starting download...")
    
    # Use multiple threads to download 15 images simultaneously
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(download_image, urls), total=len(urls), desc="Downloading Images"))
    
    success = sum(results)
    print(f"\n✅ Successfully downloaded: {success} images.")
    print(f"❌ Failed/Skipped: {len(urls) - success}")
    print(f"📁 Images saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()