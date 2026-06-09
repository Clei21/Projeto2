import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

SPIDER_URL = "https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2w9xBXWSpJadg7kXhj8ntn&export=download&confirm=t"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "spider.zip"
    out_dir = data_dir / "spider"

    if out_dir.exists():
        print(f"Spider already exists at {out_dir}")
        return

    print("Downloading Spider. If Google blocks the download, upload spider.zip manually to data/spider.zip")
    urlretrieve(SPIDER_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print("Done")


if __name__ == "__main__":
    main()
