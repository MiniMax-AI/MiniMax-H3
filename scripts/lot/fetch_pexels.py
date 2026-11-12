#!/usr/bin/env python3
"""Download people photos with captions for two-track training: the ``gaunernst/pexels-portrait`` list.

    python3 lot/fetch_pexels.py --out /data/tt_people --count 2000 [--orientation landscape]

The Hugging Face dataset ``gaunernst/pexels-portrait`` (9,853 rows) is a CSV of Pexels photo URLs,
sizes and one-sentence ``alt`` descriptions; it ships no images. This script reads that CSV, keeps
photos in the requested orientation, and downloads each at ``--width`` pixels wide (Pexels' CDN
resizes on request) with a same-stem ``.txt`` caption. The folder is then input for
``build_cache_h3.py media``.

Why this set: the faces that fail are small faces in a scene (a person on a sofa at 1344x768), and
these are photographs of people at many framings, at high resolution, with captions. Close-up face
sets (FFHQ, CelebA-HQ) make the face fill the frame, which the LR track already handles.

License: photos are under the Pexels license (free to use, no attribution required; do not
redistribute them as a stock collection). Check it for your use and keep the downloads private.
Resumable: existing files are skipped.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import time
import urllib.request
from pathlib import Path

REPO = "gaunernst/pexels-portrait"
FILE = "pexels_portrait_part1.csv.gz"


def rows(orientation: str, min_side: int):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            width, height = int(row["width"]), int(row["height"])
            if min(width, height) < min_side or not row["alt"].strip():
                continue
            if orientation == "landscape" and width <= height:
                continue
            if orientation == "portrait" and height <= width:
                continue
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--orientation", choices=("landscape", "portrait", "any"), default="landscape",
                        help="landscape matches the 1344x768 canvas; portrait pairs with --size 768 1344")
    parser.add_argument("--width", type=int, default=2048, help="download width in pixels (CDN resize)")
    parser.add_argument("--min-side", type=int, default=1024)
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds between downloads")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for row in rows(args.orientation, args.min_side):
        if done >= args.count:
            break
        stem = f"pexels_{row['id']}"
        image, caption = args.out / f"{stem}.jpg", args.out / f"{stem}.txt"
        if image.is_file() and caption.is_file():
            done += 1
            continue
        url = f"{row['src']}?auto=compress&cs=tinysrgb&w={args.width}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mpeg-diffusion-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
        except Exception as error:  # noqa: BLE001 - one bad URL should not stop the run
            failed += 1
            print(f"LOT_H3 kind=fetch_fail id={row['id']} error={error}", flush=True)
            continue
        image.write_bytes(data)
        caption.write_text(row["alt"].strip() + "\n")
        done += 1
        if done % 100 == 0:
            print(f"LOT_H3 kind=fetch done={done} failed={failed}", flush=True)
        time.sleep(args.sleep)
    print(f"LOT_H3 kind=fetch_done out={args.out} images={done} failed={failed} ok=1", flush=True)


if __name__ == "__main__":
    main()
