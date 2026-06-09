import argparse
import os
import zipfile

from scripts.config import DATA_DIR, SPIDER_DIR


def download_via_hf():
    os.makedirs(SPIDER_DIR, exist_ok=True)
    db_path = os.path.join(SPIDER_DIR, "database/<db_id>/<db_id>.sqlite")
    print(
        "Download the Spider archive (the HF dataset omits SQLite files).\n"
        "Place the following files under the data directory:\n"
        f"  {os.path.join(SPIDER_DIR, 'train_spider.json')}\n"
        f"  {os.path.join(SPIDER_DIR, 'dev.json')}\n"
        f"  {os.path.join(SPIDER_DIR, 'tables.json')}\n"
        f"  {db_path}\n"
    )


def unzip_archive(zip_path: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(DATA_DIR)
    print(f"Extracted {zip_path} into {DATA_DIR}")
    print(f"Ensure the contents are reachable under {SPIDER_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", default=None)
    args = parser.parse_args()

    if args.zip_path:
        unzip_archive(args.zip_path)
    else:
        download_via_hf()


if __name__ == "__main__":
    main()
