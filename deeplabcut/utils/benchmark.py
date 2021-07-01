import os
import urllib.request
import zipfile
from io import BytesIO
from tqdm import tqdm


def unzip_from_url(url, dest_folder):
    """Directly extract files without writing the archive to disk."""
    os.makedirs(dest_folder, exist_ok=True)
    resp = urllib.request.urlopen(url)
    with zipfile.ZipFile(BytesIO(resp.read())) as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            try:
                zf.extract(member, path=dest_folder)
            except zipfile.error:
                pass


def get_example_datasets(dest_folder="."):
    loc = os.path.join(dest_folder, "DLCexampleprojects-main")
    if not os.path.isdir(loc):
        url = "https://github.com/DeepLabCut/DLCexampleprojects/archive/main.zip"
        unzip_from_url(url, dest_folder)
    return loc
