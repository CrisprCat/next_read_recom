#!/usr/bin/env python3

"""
Download a file over HTTP(S), FTP, or copy from a local path.
"""

import argparse
import os
import sys
from urllib.parse import urlparse
import requests
from tqdm import tqdm


def download_http_ftp(url, output_filename):
    """Download a file from HTTP or FTP."""
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 8192  # 8 KB chunks

        with open(output_filename, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=output_filename,
            initial=0,
            ascii=True
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=block_size):
                f.write(chunk)
                progress_bar.update(len(chunk))


def copy_local_file(src_path, output_filename):
    """Copy a local file to the destination with a progress bar."""
    total_size = os.path.getsize(src_path)
    block_size = 8192

    with open(src_path, 'rb') as src, open(output_filename, 'wb') as dst, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc=output_filename,
        initial=0,
        ascii=True
    ) as progress_bar:
        while True:
            chunk = src.read(block_size)
            if not chunk:
                break
            dst.write(chunk)
            progress_bar.update(len(chunk))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download a file over HTTP(S), FTP, or copy from a local path."
    )
    parser.add_argument(
        "source",
        help="Source URL or local file path (http://, https://, ftp://, or /local/path)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Optional output filename. Defaults to source's filename."
    )

    args = parser.parse_args()

    # Determine filename
    if args.output:
        output_filename = args.output
    else:
        parsed = urlparse(args.source)
        output_filename = os.path.basename(parsed.path) or "downloaded_file"

    # Detect source type
    if args.source.startswith(("http://", "https://", "ftp://")):
        try:
            download_http_ftp(args.source, output_filename)
            print(f"\n✅ Download complete: {output_filename}")
        except Exception as e:
            print(f"❌ Error downloading from URL: {e}")
            sys.exit(1)

    elif os.path.exists(args.source):
        try:
            copy_local_file(args.source, output_filename)
            print(f"\n✅ File copied: {output_filename}")
        except Exception as e:
            print(f"❌ Error copying local file: {e}")
            sys.exit(1)

    else:
        print("❌ Invalid source. Must be an existing local file or a valid URL (http, https, ftp).")
        sys.exit(1)


if __name__ == "__main__":
    main()
