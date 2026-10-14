"""Build the rx schema from the downloaded Part D files.

Steps:
  1. Read 3 years of prescriber x drug rows.
  2. Design sales territories: k-means on city coordinates, weighted by how
     many prescribers practise in each city, so territories follow where the
     HCPs actually are rather than raw geography.
  3. Build the drug dimension and attach WHO ATC classes (= markets).
  4. Bulk-load with COPY, add indexes, apply row-level security.
  5. Create one Postgres login per territory manager, plus a national login.

Usage:  python scripts/load_rx.py            (state defaults to CT)
"""
import io
import os
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
load_dotenv(ROOT / ".env")

STATE = sys.argv[1] if len(sys.argv) > 1 else "CT"
N_TERRITORIES = 6

# City spellings in the prescriber file that differ from the geocoding table.
CITY_ALIASES = {
    "Vernon Rockville": "Vernon",
    "Storrs Mansfield": "Storrs",
    "W Hartford": "West Hartford",
    "S Woodstock": "Woodstock",
    "New Preston Marble Dale": "New Preston",
}
MANUAL_COORDS = {"East Lyme": (41.3568, -72.2295)}  # missing from the lookup table

ADMIN_DB = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "sslmode": os.getenv("DB_SSLMODE", "prefer"),
}
ROLE_PASSWORD = os.getenv("RX_ROLE_PASSWORD")


def read_prescriptions() -> pd.DataFrame:
    files = sorted(RAW.glob(f"partd_{STATE}_*.parquet"))
    if not files:
        raise SystemExit(f"No data in {RAW}. Run scripts/download_data.py {STATE} first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.rename(columns={
        "Year": "year", "City": "city", "Prescriber NPI": "npi",
        "Prescriber Name": "prescriber_name", "Specialty": "specialty",
        "Brand Name": "brand_name", "Generic Name": "generic_name",
        "Total Claims": "total_claims", "Total 30-Day Fills": "total_30day_fills",
        "Total Days Supply": "total_days_supply", "Total Drug Cost": "total_drug_cost",
        "Total Beneficiaries": "total_beneficiaries",
    })


def build_prescribers(df: pd.DataFrame) -> pd.DataFrame:
    # A few providers moved city or changed specialty; keep the latest year.
    latest = df.sort_values("year").groupby("npi").tail(1)
    return latest[["npi", "prescriber_name", "specialty", "city"]].reset_index(drop=True)


def design_territories(prescribers: pd.DataFrame):
    cities = pd.read_parquet(RAW / "cities.parquet")
    coords = cities[cities.State == STATE].set_index("City")[["Latitude", "Longitude"]]
    for city, (lat, lon) in MANUAL_COORDS.items():
        coords.loc[city] = (lat, lon)

    geo_city = prescribers["city"].replace(CITY_ALIASES)
    located = prescribers.assign(geo_city=geo_city).join(coords, on="geo_city")
    has_coords = located["Latitude"].notna()

    # One point per city, weighted by its prescriber count.
    per_city = (located[has_coords]
                .groupby("geo_city")
                .agg(lat=("Latitude", "first"), lon=("Longitude", "first"), hcps=("npi", "size")))
    km = KMeans(n_clusters=N_TERRITORIES, n_init=10, random_state=42)
    per_city["cluster"] = km.fit_predict(per_city[["lat", "lon"]], sample_weight=per_city["hcps"])

    # Name each territory after its biggest city by prescriber count.
    territories, cluster_to_id = [], {}
    for cluster, grp in per_city.groupby("cluster"):
        anchor = grp["hcps"].idxmax()
        tid = "T-" + re.sub(r"[^A-Z]", "", anchor.upper())[:10]
        cluster_to_id[cluster] = tid
        territories.append({"territory_id": tid, "territory_name": anchor,
                            "center_lat": round(km.cluster_centers_[cluster][0], 5),
                            "center_lon": round(km.cluster_centers_[cluster][1], 5)})
    territories.append({"territory_id": "UNALIGNED", "territory_name": "Unaligned (city not geocoded)",
                        "center_lat": None, "center_lon": None})

    city_to_tid = per_city["cluster"].map(cluster_to_id)
    prescribers["territory_id"] = located["geo_city"].map(city_to_tid).fillna("UNALIGNED")
    return pd.DataFrame(territories), prescribers


def build_drugs(df: pd.DataFrame) -> pd.DataFrame:
    drugs = df[["brand_name", "generic_name"]].drop_duplicates().sort_values(["generic_name", "brand_name"])
    drugs = drugs.reset_index(drop=True)
    drugs.insert(0, "drug_id", drugs.index + 1)
    drugs["sold_as_generic"] = drugs["brand_name"] == drugs["generic_name"]

    atc = pd.read_parquet(RAW / "drug_atc.parquet").set_index("Generic Name")[
        ["atc_level_4_code", "atc_level_4_name", "atc_level_3_name", "atc_level_1_name"]]
    atc.columns = ["atc_class_code", "atc_class_name", "atc_group_name", "therapeutic_area"]
    return drugs.join(atc, on="generic_name")


def copy_df(cur, df: pd.DataFrame, table: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cols = ", ".join(df.columns)
    cur.copy_expert(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)
    print(f"  {table:<22} {len(df):>9,} rows")


def role_name(territory_name: str) -> str:
    return "mgr_" + re.sub(r"[^a-z]+", "_", territory_name.lower()).strip("_")


def create_logins(cur, territories: pd.DataFrame):
    if not ROLE_PASSWORD:
        raise SystemExit("RX_ROLE_PASSWORD is not set in .env")
    logins = {role_name(t.territory_name): [t.territory_id]
              for t in territories.itertuples() if t.territory_id != "UNALIGNED"}
    logins["rx_national"] = territories["territory_id"].tolist()

    for role, tids in logins.items():
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
        verb = "ALTER" if cur.fetchone() else "CREATE"
        cur.execute(sql.SQL(verb + " ROLE {} LOGIN PASSWORD {}").format(
            sql.Identifier(role), sql.Literal(ROLE_PASSWORD)))
        cur.execute(sql.SQL("GRANT rx_reader TO {}").format(sql.Identifier(role)))
        # Guardrails that live on the role, so no client can forget them.
        for setting, value in (("default_transaction_read_only", "on"),
                               ("statement_timeout", "15s"),
                               ("search_path", "rx")):
            cur.execute(sql.SQL("ALTER ROLE {} SET {} = {}").format(
                sql.Identifier(role), sql.Identifier(setting), sql.Literal(value)))
        cur.executemany("INSERT INTO rx.user_territories VALUES (%s, %s)", [(role, t) for t in tids])
        print(f"  login {role:<20} -> {', '.join(tids)}")


def main():
    print(f"Reading Part D data for {STATE}...")
    df = read_prescriptions()
    prescribers = build_prescribers(df)
    territories, prescribers = design_territories(prescribers)
    drugs = build_drugs(df)

    facts = (df.merge(drugs[["drug_id", "brand_name", "generic_name"]], on=["brand_name", "generic_name"])
               .merge(prescribers[["npi", "territory_id"]], on="npi"))
    facts = facts[["year", "npi", "drug_id", "territory_id", "total_claims", "total_30day_fills",
                   "total_days_supply", "total_drug_cost", "total_beneficiaries"]]
    facts["total_beneficiaries"] = facts["total_beneficiaries"].astype("Int64")

    print(territories.merge(prescribers.territory_id.value_counts().rename("hcps"),
                            left_on="territory_id", right_index=True)[["territory_id", "territory_name", "hcps"]]
          .to_string(index=False))

    conn = psycopg2.connect(**ADMIN_DB)
    try:
        with conn, conn.cursor() as cur:
            print("Creating schema...")
            cur.execute((ROOT / "db" / "schema.sql").read_text(encoding="utf-8"))
            copy_df(cur, territories, "rx.territories")
            copy_df(cur, prescribers, "rx.prescribers")
            copy_df(cur, drugs, "rx.drugs")
            copy_df(cur, facts, "rx.prescriptions")

            print("Indexing...")
            cur.execute("""
                CREATE INDEX ON rx.prescriptions (territory_id);
                CREATE INDEX ON rx.prescriptions (drug_id, year);
                CREATE INDEX ON rx.prescriptions (npi);
                CREATE INDEX ON rx.prescribers (territory_id);
                CREATE INDEX ON rx.drugs (atc_class_code);
                CREATE INDEX ON rx.drugs (lower(brand_name));
            """)
            print("Applying row-level security...")
            cur.execute((ROOT / "db" / "security.sql").read_text(encoding="utf-8"))
            create_logins(cur, territories)
        with conn.cursor() as cur:
            conn.autocommit = True
            cur.execute("ANALYZE rx.prescriptions; ANALYZE rx.prescribers; ANALYZE rx.drugs;")
    finally:
        conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
