"""
Download real-world malaria incidence data from Our World in Data (WHO Global Health Observatory).

Dataset: "Incidence of malaria (per 1,000 population at risk)"
Source:  World Health Organization (Global Health Observatory), via World Bank
URL:     https://ourworldindata.org/grapher/incidence-of-malaria
License: CC BY 4.0 (Our World in Data); original WHO data public domain

This script downloads the full CSV and saves a Mali subset to data/real/.
"""

from __future__ import annotations

import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data", "real")
os.makedirs(DATA_DIR, exist_ok=True)

FULL_CSV_URL = (
    "https://ourworldindata.org/grapher/incidence-of-malaria.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)

MALI_CSV = os.path.join(DATA_DIR, "mali_malaria_incidence_who_2000_2024.csv")
FULL_CSV = os.path.join(DATA_DIR, "full_malaria_incidence_owid.csv")

HEADERS = {"User-Agent": "MalariaModel/1.0 (research; python)"}


def download() -> None:
    print("Downloading WHO malaria incidence data from Our World in Data ...")
    req = urllib.request.Request(FULL_CSV_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    with open(FULL_CSV, "wb") as f:
        f.write(raw)
    print(f"  Full dataset saved: {FULL_CSV} ({len(raw):,} bytes)")

    import csv
    rows = []
    with open(FULL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Code"] == "MLI":
                rows.append(row)

    if not rows:
        print("  WARNING: no Mali rows found!")
        return

    fieldnames = ["Entity", "Code", "Year",
                  "Incidence of malaria (per 1,000 population at risk)"]
    with open(MALI_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Mali subset saved: {MALI_CSV} ({len(rows)} years)")
    print(f"  Years: {rows[0]['Year']} - {rows[-1]['Year']}")
    print(f"  Incidence range: {min(float(r[fieldnames[3]]) for r in rows):.1f} - "
          f"{max(float(r[fieldnames[3]]) for r in rows):.1f} per 1,000 at risk")


if __name__ == "__main__":
    download()
