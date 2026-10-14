"""Download CMS Medicare Part D prescriber data for one state.

Source: CMS "Medicare Part D Prescribers - by Provider and Drug" (public,
US-government work). data.cms.gov blocks many non-US IPs, so this pulls the
same data from a Parquet mirror on Hugging Face that is partitioned by
year and state, plus two lookup tables from the same mirror:
  - cities.parquet   : city -> latitude/longitude (used to design territories)
  - drug_atc.parquet : generic name -> WHO ATC therapeutic class (defines "markets")

Usage:  python scripts/download_data.py            (defaults to CT, 2021-2023)
        python scripts/download_data.py NJ 2022 2023
"""
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://huggingface.co/datasets/mbateya/medicare_part_d_prescribers/resolve/main"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch(url: str, dest: Path):
    if dest.exists():
        print(f"  exists  {dest.name}")
        return
    print(f"  fetch   {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main():
    state = sys.argv[1] if len(sys.argv) > 1 else "CT"
    years = [int(y) for y in sys.argv[2:]] or [2021, 2022, 2023]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for year in years:
        fetch(f"{MIRROR}/prescribers/year={year}/State={state}/data_0.parquet",
              RAW_DIR / f"partd_{state}_{year}.parquet")
    for name in ("cities", "drug_atc"):
        fetch(f"{MIRROR}/{name}.parquet", RAW_DIR / f"{name}.parquet")
    print("done")


if __name__ == "__main__":
    main()
