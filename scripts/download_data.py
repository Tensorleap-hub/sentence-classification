"""Download and unzip the public Kaggle dataset (no account required).

The Kaggle dataset endpoint 302-redirects to a signed public URL serving the
archive, so anonymous download works.
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
URL = "https://www.kaggle.com/api/v1/datasets/download/msd23004/final-classification-dataset"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as resp:  # noqa: S310 (trusted host)
        data = resp.read()
    print(f"downloaded {len(data)} bytes")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(RAW)
        names = zf.namelist()
    print(f"extracted to {RAW}: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
