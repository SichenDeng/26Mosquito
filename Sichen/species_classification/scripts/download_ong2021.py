"""Download the unrotated images (the _0 and _0_testing folders) of the Ong et al. 2021 Aedes dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/pradeepisawasan/aedes-mosquitos (CC BY-NC-SA 4.0)
The _90/_180/_270 folders are rotated copies of the same photos, so we skip them
and do rotation as augmentation at training time instead.

Token: set the environment variable KAGGLE_API_TOKEN, or save the token to ~/.kaggle/access_token.
Usage:  python download_ong2021.py [output_dir]   (default: ../data/raw/ong2021)
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "pradeepisawasan/aedes-mosquitos"
API = "https://www.kaggle.com/api/v1/datasets"
KEEP_FOLDERS = {"aegypti_0", "aegypti_0_testing", "albopictus_0", "albopictus_0_testing"}
WORKERS = 2  # more parallel requests trigger Kaggle's rate limit (HTTP 429)
MAX_RETRIES = 6


def load_token():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        token_file = Path.home() / ".kaggle" / "access_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        sys.exit("No token found: set KAGGLE_API_TOKEN or create ~/.kaggle/access_token")
    if not token.isascii() or not token.startswith("KGAT_"):
        sys.exit("Token looks wrong: it should be the real token starting with KGAT_, letters/digits only")
    return token


def list_files():
    """List every file in the dataset (the API returns them one page at a time)."""
    names, page_token = [], None
    while True:
        query = {"pageSize": "1000"}
        if page_token:
            query["pageToken"] = page_token
        url = f"{API}/list/{DATASET}?" + urllib.parse.urlencode(query)
        data = json.load(urllib.request.urlopen(url))
        names += [f["name"] for f in data.get("datasetFiles", [])]
        page_token = data.get("nextPageTokenNullable")
        if not page_token:
            return names


def download_one(name, out_dir, token):
    # Keep only the last two path levels, e.g. aegypti_0/aegypti100a.jpg
    dest = out_dir / Path(*Path(name).parts[-2:])
    if dest.exists():
        return "skip"
    url = f"{API}/download/{DATASET}/" + urllib.parse.quote(name, safe="")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    for attempt in range(MAX_RETRIES):
        try:
            payload = urllib.request.urlopen(req, timeout=120).read()
            break
        except urllib.error.HTTPError as e:
            # 429 = Kaggle rate limit: wait (doubling each time) and try again.
            # Kaggle's Retry-After (5-10 s) is often too short, so never wait less than the backoff.
            if e.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            wait = max(int(e.headers.get("Retry-After") or 0), 30 * 2 ** attempt)
            print(f"rate limited, waiting {wait}s ...", flush=True)
            time.sleep(wait)
    # Kaggle wraps each single file in a zip archive
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        data = zf.read(zf.namelist()[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return "ok"


def main():
    default_out = Path(__file__).resolve().parent.parent / "data" / "raw" / "ong2021"
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_out
    token = load_token()

    wanted = [n for n in list_files() if Path(n).parent.name in KEEP_FOLDERS]
    print(f"{len(wanted)} files to download into {out_dir}")

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for status in pool.map(lambda n: download_one(n, out_dir, token), wanted):
            done += 1
            if done % 100 == 0 or done == len(wanted):
                print(f"{done}/{len(wanted)}", flush=True)


if __name__ == "__main__":
    main()
