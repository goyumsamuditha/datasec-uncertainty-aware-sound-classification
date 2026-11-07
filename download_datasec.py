"""
Download DataSEC v4 from its official Zenodo record.

The archive is saved outside the repository by default:
    <project parent>/DATASEC/DATASEC.zip

Usage:
    python download_datasec.py
    python download_datasec.py --output-dir "D:/Datasets/DataSEC"
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RECORD_ID = "23097290"
FILE_NAME = "DATASEC.zip"
EXPECTED_MD5 = "11e0ae0098ecf3d982ab5f7375c56844"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK_SIZE = 8 * 1024 * 1024


def get_record_metadata():
    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "DataSEC-Downloader/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def calculate_md5(path):
    digest = hashlib.md5()

    with path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def download_file(url, destination, expected_size):
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0

    headers = {"User-Agent": "DataSEC-Downloader/1.0"}

    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    request = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416 and partial.exists():
            print("Checking existing partial download...")
            return partial
        raise

    with response:
        if downloaded and response.status == 206:
            mode = "ab"
        else:
            mode = "wb"
            downloaded = 0

        total = expected_size or 0

        print(f"Downloading {FILE_NAME}...")
        print(f"Destination: {destination}")
        if total:
            print(f"Expected size: {total / (1024 ** 3):.2f} GB")

        with partial.open(mode) as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded / total * 100
                    print(
                        f"\rProgress: {percent:.1f}% "
                        f"({downloaded / (1024 ** 3):.2f} GB)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rDownloaded: {downloaded / (1024 ** 3):.2f} GB",
                        end="",
                        flush=True,
                    )

    print()
    return partial


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify the DataSEC v4 dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "DATASEC",
        help="Folder where DATASEC.zip will be saved.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / FILE_NAME

    print("Retrieving official Zenodo record metadata...")

    try:
        metadata = get_record_metadata()

        files = metadata.get("files", [])
        dataset_file = next(
            (item for item in files if item.get("key") == FILE_NAME),
            None,
        )

        if dataset_file is None:
            raise RuntimeError(
                f"{FILE_NAME} was not found in Zenodo record {RECORD_ID}."
            )

        checksum = dataset_file.get("checksum", "")
        if checksum.lower() != f"md5:{EXPECTED_MD5}":
            raise RuntimeError(
                "The checksum in Zenodo metadata does not match "
                "the expected DataSEC v4 checksum. "
                "Please check the official record before downloading."
            )

        file_size = dataset_file.get("size")
        download_url = dataset_file.get("links", {}).get("content")

        if not download_url:
            raise RuntimeError(
                "Zenodo did not provide a file download URL."
            )

        if destination.exists():
            print("An archive already exists. Checking its checksum...")
            if calculate_md5(destination) == EXPECTED_MD5:
                print("Existing archive is valid. No download needed.")
                return
            print("Existing archive is invalid and will be replaced.")
            destination.unlink()

        partial = download_file(
            download_url,
            destination,
            file_size,
        )

        print("Verifying MD5 checksum...")
        actual_md5 = calculate_md5(partial)

        if actual_md5 != EXPECTED_MD5:
            raise RuntimeError(
                "Checksum verification failed.\n"
                f"Expected: {EXPECTED_MD5}\n"
                f"Actual:   {actual_md5}\n"
                "The partial file has been retained for inspection."
            )

        partial.replace(destination)

        print("\nDownload and verification completed successfully.")
        print(f"Archive: {destination}")
        print(f"MD5: {actual_md5}")
        print("The archive has not been extracted.")

    except Exception as error:
        print(f"\nDownload failed: {error}", file=sys.stderr)
        print(
            "If a partial download exists, run the script again "
            "to attempt to resume it.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
