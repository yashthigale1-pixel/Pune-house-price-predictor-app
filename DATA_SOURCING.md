# Data Sourcing & Methodology — Pune Housing Dataset

**This dataset is SYNTHETIC. It is not a record of real transactions.**
No reliable, freely-downloadable transaction-level dataset for Pune real
estate exists in the public domain (unlike the U.S. census housing data).
Rather than use a fake dataset mislabeled as real (a common problem with
"Pune house price" files circulating online — several are just the
Bangalore dataset with location names swapped), this dataset is generated
programmatically using a documented price model, calibrated to real,
sourced 2026 per-sq-ft benchmarks for actual Pune localities.

## Locality price-per-sqft benchmarks used (₹/sq ft, 2026)

Sourced from NoBroker, 99acres, and Pune real-estate market reports
(July 2026):

| Zone | Localities | ₹/sq ft range |
|---|---|---|
| Premium West | Baner, Aundh, Kalyani Nagar, Koregaon Park, Boat Club Road, Pashan | 12,000–22,000 |
| IT Corridor / Upper-mid | Kharadi, Viman Nagar, Kothrud, Balewadi | 9,000–16,000 |
| Mid-tier | Hinjewadi, Wakad, Bavdhan, Ravet, Magarpatta | 7,000–10,000 |
| Central affordable | Camp, Narayan Peth, Somwar Peth, Swargate, Shivaji Nagar | 6,000–9,000 |
| Outskirts / affordable | Wagholi, Moshi, Undri, Narhe, Kondhwa, Hadapsar, Chinchwad, Kiwale, NIBM Road, Sinhgad Road | 5,000–8,000 |

Sources:
- https://www.nobroker.in/blog/property-rates-in-pune/
- https://www.99acres.com/property-rates-and-price-trends-in-pune-prffid
- https://www.99acres.com/property-rates-and-price-trends-in-baner-pune-prffid
- https://www.99acres.com/property-rates-and-price-trends-in-hinjewadi-pune-prffid
- https://bankinginputs.com/real-estate-pune/
- https://buyinpune.com/guides/baner

## Price model (how each row is generated)

For each synthetic listing:

1. A locality is sampled, giving a base ₹/sqft drawn from that locality's
   real benchmark range (± random noise for within-locality variation —
   real listings in the same area vary due to building quality, exact
   micro-location, etc.).
2. `total_sqft` is sampled conditional on BHK count (larger configurations
   → larger area, with realistic overlap).
3. Adjustment factors are applied on top of the base rate, each reflecting
   a well-documented real-estate pricing pattern:
   - **Age**: newer construction (0–2 yrs) commands a premium; price
     depreciates gradually up to ~20+ years.
   - **Floor**: mid-to-high floors carry a small view/ventilation premium
     in Indian apartment pricing; ground floor is discounted.
   - **Property type**: villas > apartments > builder floors, per sq ft.
   - **Amenities score** (clubhouse, gym, security, lift, power backup):
     more amenities → modest premium, consistent with premium project
     pricing patterns cited above.
   - **BHK**: larger configurations see a small per-sqft premium
     (common in Indian listings, since bigger units are often in newer/
     premium projects).
4. Gaussian noise is added to simulate real-world listing variance
   (negotiation, seller pricing behavior, condition, etc.).

## What this dataset is useful for

- Practicing an end-to-end regression pipeline with realistic feature
  relationships and non-trivial noise.
- Demonstrating feature engineering (price-per-sqft, locality encoding,
  zone grouping) on India-specific real estate structure.

## What it is NOT useful for

- Real Pune property valuation or investment decisions.
- Any claim about actual transacted prices — these are illustrative
  numbers calibrated to public market-rate ranges, not verified sales.
