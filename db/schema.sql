-- ─────────────────────────────────────────────────────────────────────
-- Pharma commercial analytics schema, built on CMS Medicare Part D data.
-- Lives in its own schema (rx) so it never collides with anything in public.
--
-- The COMMENT ON statements are not decoration: the agent reads them from
-- the catalog and puts them in the LLM prompt. Documenting a column here
-- is how you teach the model what it means.
-- ─────────────────────────────────────────────────────────────────────
DROP SCHEMA IF EXISTS rx CASCADE;
CREATE SCHEMA rx;

CREATE TABLE rx.territories (
    territory_id   TEXT PRIMARY KEY,
    territory_name TEXT NOT NULL,
    center_lat     NUMERIC(8,5),
    center_lon     NUMERIC(8,5)
);
COMMENT ON TABLE rx.territories IS
  'Sales territories. Designed by k-means clustering of prescriber locations; named after the largest city in each cluster. UNALIGNED holds prescribers whose city could not be geocoded.';

CREATE TABLE rx.prescribers (
    npi             TEXT PRIMARY KEY,
    prescriber_name TEXT NOT NULL,
    specialty       TEXT,
    city            TEXT,
    territory_id    TEXT NOT NULL REFERENCES rx.territories(territory_id)
);
COMMENT ON TABLE  rx.prescribers IS 'Healthcare providers (HCPs) who prescribed to Medicare Part D patients. One row per provider; city and specialty are from their most recent year.';
COMMENT ON COLUMN rx.prescribers.npi IS 'National Provider Identifier, the unique ID of an HCP.';
COMMENT ON COLUMN rx.prescribers.specialty IS 'Provider specialty as reported to CMS, e.g. Cardiology, Internal Medicine, Nurse Practitioner, Family Practice, Endocrinology.';

CREATE TABLE rx.drugs (
    drug_id          INTEGER PRIMARY KEY,
    brand_name       TEXT NOT NULL,
    generic_name     TEXT NOT NULL,
    sold_as_generic  BOOLEAN NOT NULL,
    atc_class_code   TEXT,
    atc_class_name   TEXT,
    atc_group_name   TEXT,
    therapeutic_area TEXT,
    UNIQUE (brand_name, generic_name)
);
COMMENT ON TABLE  rx.drugs IS 'Drug products. WHO ATC codes classify each drug; a "market" is all drugs sharing an atc_class_code.';
COMMENT ON COLUMN rx.drugs.brand_name IS 'Brand name as dispensed, e.g. Eliquis, Ozempic, Jardiance. Equals generic_name when a generic was dispensed.';
COMMENT ON COLUMN rx.drugs.sold_as_generic IS 'TRUE when brand_name = generic_name, i.e. the product was dispensed as a generic.';
COMMENT ON COLUMN rx.drugs.atc_class_code IS 'ATC level-4 code = the competitive market. Filter markets by code, never by keywords in atc_class_name. Anticoagulants/DOACs = B01AF + B01AE (Eliquis, Xarelto, Savaysa, Pradaxa; add B01AA, B01AB only if warfarin or heparin are asked for); GLP-1 = A10BJ (Ozempic, Trulicity, Rybelsus); SGLT2 = A10BK (Jardiance, Farxiga, Invokana); DPP-4 = A10BH (Januvia); statins = C10AA; LAMA/LABA inhalers = R03AL. For any other brand, look up its code in drugs.';
COMMENT ON COLUMN rx.drugs.atc_class_name IS 'ATC level-4 name, e.g. "Direct factor Xa inhibitors".';
COMMENT ON COLUMN rx.drugs.atc_group_name IS 'ATC level-3 name, a broader group, e.g. "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS".';
COMMENT ON COLUMN rx.drugs.therapeutic_area IS 'ATC level-1 name, e.g. "CARDIOVASCULAR SYSTEM".';

CREATE TABLE rx.prescriptions (
    year                SMALLINT NOT NULL,
    npi                 TEXT     NOT NULL REFERENCES rx.prescribers(npi),
    drug_id             INTEGER  NOT NULL REFERENCES rx.drugs(drug_id),
    territory_id        TEXT     NOT NULL REFERENCES rx.territories(territory_id),
    total_claims        INTEGER  NOT NULL,
    total_30day_fills   NUMERIC(12,1),
    total_days_supply   INTEGER,
    total_drug_cost     NUMERIC(14,2),
    total_beneficiaries INTEGER,
    PRIMARY KEY (year, npi, drug_id)
);
COMMENT ON TABLE  rx.prescriptions IS 'Annual prescribing volume per prescriber per drug (2021-2023). CMS omits rows with fewer than 11 claims.';
COMMENT ON COLUMN rx.prescriptions.year IS 'Calendar year: 2021, 2022 or 2023 (annual totals only). If the question names no year and is not about a trend, filter to 2023, the latest year; never silently sum several years.';
COMMENT ON COLUMN rx.prescriptions.total_claims IS 'Number of Part D prescriptions filled, including refills. The standard volume measure (proxy for TRx).';
COMMENT ON COLUMN rx.prescriptions.total_30day_fills IS 'Claims normalised to 30-day supplies, so a 90-day fill counts as 3. Use for fair volume comparison across pack sizes.';
COMMENT ON COLUMN rx.prescriptions.total_drug_cost IS 'Total amount paid for the drug in US dollars (all payers: plan, patient, subsidies).';
COMMENT ON COLUMN rx.prescriptions.total_beneficiaries IS 'Distinct patients. NULL when CMS suppressed it for privacy (1-10 patients). Never SUM this across rows: patients overlap between drugs and prescribers.';
COMMENT ON COLUMN rx.prescriptions.territory_id IS 'Territory of the prescriber, denormalised here so row-level security can filter this large table without a join.';

-- Maps a database login to the territories it may see. Read by the RLS policies.
CREATE TABLE rx.user_territories (
    db_role      TEXT NOT NULL,
    territory_id TEXT NOT NULL REFERENCES rx.territories(territory_id),
    PRIMARY KEY (db_role, territory_id)
);
COMMENT ON TABLE rx.user_territories IS 'Which territories each database login is allowed to see. Only your own rows are visible.';
