import urllib.request
import zipfile
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_synthtext(output_dir=None):
    """Download SynthText dataset from official source."""
    if output_dir is None:
        output_dir = config.RAW_DATA_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / 'SynthText.zip'

    # Check if already downloaded
    if config.SYNTHTEXT_MAT.exists():
        print("SynthText dataset already exists.")
        return

    if zip_path.exists():
        print("Zip file exists, extracting...")
    else:
        print(f"Downloading SynthText dataset from {config.SYNTHTEXT_URL}")
        print("Note: This is a large file (~41GB), download may take a while...")

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Downloading') as t:
            urllib.request.urlretrieve(config.SYNTHTEXT_URL, zip_path, reporthook=t.update_to)

        print(f"Download complete: {zip_path}")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Extraction complete. Dataset available at: {output_dir / 'SynthText'}")

    # Optionally remove zip file
    # zip_path.unlink()

if __name__ == '__main__':
    download_synthtext()
