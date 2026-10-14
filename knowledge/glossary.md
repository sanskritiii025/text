# Pharma commercial analytics glossary

Each "## " section is one retrievable entry. The agent retrieves the most
relevant entries for a question and puts them in the prompt, so every
section should stand on its own. SQL patterns here are the approved way to
compute a metric; the agent is told to follow them.

## Prescription volume (claims, TRx)
"Prescriptions", "scripts", "Rx", "TRx" and "volume" all mean SUM(total_claims) from prescriptions.
A claim is one filled prescription including refills, which makes it the closest proxy for TRx (total prescriptions).
NRx (new prescriptions) is NOT available: CMS does not split new starts from refills. If asked for NRx or new patients, say so and offer TRx instead.
For volume that is fair across 30-day and 90-day packs, use SUM(total_30day_fills).

## Drug cost and cost per claim
"Spend", "sales", "revenue" and "cost" mean SUM(total_drug_cost), in US dollars, paid by all payers for Medicare patients.
Cost per claim must be a ratio of sums, never an average of row-level ratios:
SELECT d.brand_name, SUM(p.total_drug_cost) / NULLIF(SUM(p.total_claims), 0) AS cost_per_claim
FROM prescriptions p JOIN drugs d USING (drug_id) GROUP BY d.brand_name;

## Patients and beneficiaries
total_beneficiaries is the number of distinct patients for one prescriber, drug and year.
It is NULL when 1-10 patients (privacy suppression), which is about 60% of rows.
Never SUM total_beneficiaries across prescribers or drugs: the same patient is counted in several rows, so the total is wrong. Report it only at single-row level, or say that patient counts cannot be aggregated.

## Finding a drug by name
Brand names have variants, e.g. "Humira(Cf) Pen", "Humira Pen", "Humira(Cf)". Match case-insensitively with a prefix:
WHERE lower(d.brand_name) LIKE 'humira%'
Generic names are in drugs.generic_name, e.g. 'Apixaban' for Eliquis, 'Semaglutide' for Ozempic. When the user names a molecule, match generic_name the same way.

## Market names to ATC codes (anticoagulants, diabetes, GLP-1, SGLT2, statins, inhalers)
Markets are defined by atc_class_code. NEVER search atc_class_name or atc_group_name for words like 'anticoagulant' or 'diabetes': the stored names are technical ("Direct factor Xa inhibitors") and keyword searches silently return nothing. Use these codes:
- Anticoagulants, blood thinners, DOACs, NOACs: atc_class_code IN ('B01AF','B01AE') = Eliquis, Xarelto, Savaysa, Pradaxa. "All anticoagulants including warfarin/heparin": add 'B01AA','B01AB'.
- Antiplatelets: 'B01AC' (Brilinta, clopidogrel).
- GLP-1 agonists: 'A10BJ' (Ozempic, Trulicity, Rybelsus, Byetta, Bydureon). Mounjaro (tirzepatide) is classified 'A10BX'; include it only if the user asks for incretins or names it.
- SGLT2 inhibitors: 'A10BK' (Jardiance, Farxiga, Invokana). DPP-4 inhibitors: 'A10BH' (Januvia).
- Insulins: atc_class_code LIKE 'A10A%'. All diabetes drugs: LIKE 'A10%'.
- Statins: 'C10AA' (Lipitor, Crestor and generics). All lipid-lowering: LIKE 'C10%'.
- Respiratory inhalers: LIKE 'R03%'. LAMA/LABA and triple therapy: 'R03AL' (Trelegy Ellipta, Anoro Ellipta, Stiolto Respimat).
For any other market, look up the brand's code first: SELECT DISTINCT atc_class_code, atc_class_name FROM drugs WHERE lower(brand_name) LIKE 'brandname%'.

## Market and market share
A market is the set of drugs in the relevant atc_class_code(s); see "Market names to ATC codes".
Market share of a brand = brand claims / all claims in its market, for the same year and geography.
SELECT d.brand_name, SUM(p.total_claims) AS claims,
       ROUND(100.0 * SUM(p.total_claims) / SUM(SUM(p.total_claims)) OVER (), 1) AS share_pct
FROM prescriptions p JOIN drugs d USING (drug_id)
WHERE d.atc_class_code IN ('B01AF','B01AE') AND p.year = 2023
GROUP BY d.brand_name ORDER BY claims DESC;
If the user names a brand but not a market, use the brand's market from "Market names to ATC codes", or look up its atc_class_code.

## Year-over-year growth and trends
Data covers calendar years 2021, 2022 and 2023 (annual totals, no months or quarters). "Latest year" means 2023.
DEFAULT PERIOD: if the question gives no year and is not about a trend or change over time, filter to year = 2023. Never silently sum all three years; a 3-year total is not a business-standard number.
Growth % = 100 * (this_year - last_year) / last_year. Compute it with conditional aggregation:
SELECT d.brand_name,
       SUM(p.total_claims) FILTER (WHERE p.year = 2022) AS claims_2022,
       SUM(p.total_claims) FILTER (WHERE p.year = 2023) AS claims_2023,
       ROUND(100.0 * (SUM(p.total_claims) FILTER (WHERE p.year = 2023) - SUM(p.total_claims) FILTER (WHERE p.year = 2022))
             / NULLIF(SUM(p.total_claims) FILTER (WHERE p.year = 2022), 0), 1) AS growth_pct
FROM prescriptions p JOIN drugs d USING (drug_id) GROUP BY d.brand_name;
For trend questions return one row per year so it can be charted.

## Prescriber deciles (HCP segmentation)
Deciles rank prescribers by their volume in a market. In pharma, deciles are volume-based: decile 10 is the smallest group of top prescribers who together write the top 10% of volume, so decile 10 has far fewer HCPs than decile 1. Do not use NTILE for this, it makes equal-sized groups.
WITH hcp AS (
  SELECT p.npi, SUM(p.total_claims) AS claims
  FROM prescriptions p JOIN drugs d USING (drug_id)
  WHERE d.atc_class_code = 'A10BJ' AND p.year = 2023 GROUP BY p.npi
), ranked AS (
  SELECT npi, claims, SUM(claims) OVER (ORDER BY claims, npi) / SUM(claims) OVER () AS cum_share FROM hcp
)
SELECT npi, claims, GREATEST(1, CEIL(10 * cum_share))::int AS decile FROM ranked;
"High-decile" or "top" prescribers usually means decile 8-10.

## Targeting and headroom (high market volume, low brand share)
A targeting opportunity is a prescriber who writes a lot in the market but little of our brand: high market volume, low brand share (e.g. "writes many anticoagulants but gives Eliquis less than half").
Pattern: compute per prescriber market_claims and brand_claims for the year, then brand_share = brand_claims / market_claims, and rank by market_claims among those with share below a threshold (e.g. 20%).
Prescribers with ZERO brand claims are the biggest opportunity, so keep them: count brand claims with SUM(CASE WHEN ... THEN total_claims ELSE 0 END) and never wrap that in NULLIF, which turns 0 into NULL and silently drops them from a "< threshold" filter.
Include prescriber_name, specialty and city so a rep can act on the list.

## Prescriber breadth and new writers
"Writers" or "prescribers of X" = COUNT(DISTINCT npi) with claims for the drug.
A "new writer" in 2023 is a prescriber with claims in 2023 and none in 2022 for that drug. Caveat: rows under 11 claims are suppressed, so a "new writer" may have written a few prescriptions before; call them "new or newly-above-threshold writers".
A "lapsed writer" had claims in 2022 and none in 2023.

## Brand vs generic
drugs.sold_as_generic is TRUE when the product was dispensed as a generic (brand_name = generic_name).
Generic share of a molecule = claims where sold_as_generic / all claims of that generic_name.

## Specialties and prescriber attributes
prescribers.specialty holds values like 'Internal Medicine', 'Family Practice', 'Nurse Practitioner', 'Physician Assistant', 'Cardiology', 'Endocrinology', 'Psychiatry', 'Neurology', 'Pulmonary Disease', 'Nephrology', 'Hematology-Oncology', 'Rheumatology'.
"Doctors" or "HCPs" means all prescribers, including nurse practitioners and physician assistants, unless the user says physicians only.
Match specialties case-insensitively: lower(specialty) = 'cardiology'. "Primary care" = Internal Medicine, Family Practice, General Practice, Nurse Practitioner, Physician Assistant.

## Territories and data access
Every user sees only the territories they are assigned to; the database enforces this automatically. Do not add a territory filter unless the user asks about a specific territory.
Territory names are in territories.territory_name (Hartford, New Haven, Danbury, Stamford, Waterbury, Norwich). Join prescriptions.territory_id to territories for names.
If a result is empty for a territory the user named, it may be outside their access.

## Data caveats
Source: CMS Medicare Part D Prescribers by Provider and Drug, Connecticut, 2021-2023. Covers Medicare Part D (mostly patients 65+) only, not commercial insurance or Medicaid, so it is a sample of the market, not the whole market.
Rows with fewer than 11 claims are omitted by CMS, so small prescribers and rare drugs are undercounted.
Drug cost is gross (before manufacturer rebates).
