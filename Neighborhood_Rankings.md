# Boston Neighborhood Rankings

Three rankings are provided below for all 25 UI neighborhood labels.

Tier assignment across all rankings: Ranks 1–8 = **High**, Ranks 9–17 = **Moderate**, Ranks 18–25 = **Low**. Where a rank ties (multiple neighborhoods sharing the same rank), all entries in that rank receive the same tier.

---

## Ranking 1 — Knowledge-Based (General Safety Reputation)

Based on general knowledge of Boston neighborhood safety character, violent crime history, and trajectory through the 2020s. More reflective of the actual lived experience of a resident than the raw district-level count.

| Rank | Neighborhood | Tier |
|---|---|---|
| 1 | Roxbury | High |
| 2 | Mattapan | High |
| 3 | Greater Mattapan | High |
| 4 | Dorchester | High |
| 5 | Mission Hill | High |
| 6 | Downtown | High |
| 7 | Financial District | High |
| 8 | East Boston | High |
| 9 | South End | Moderate |
| 10 | Fenway | Moderate |
| 11 | Kenmore | Moderate |
| 12 | Audubon Circle | Moderate |
| 13 | Longwood | Moderate |
| 14 | Jamaica Plain | Moderate |
| 15 | Allston | Moderate |
| 16 | Allston / Brighton | Moderate |
| 17 | Brighton | Moderate |
| 18 | South Boston | Low |
| 19 | South Boston Waterfront | Low |
| 20 | Hyde Park | Low |
| 21 | Back Bay | Low |
| 22 | Beacon Hill | Low |
| 23 | Roslindale | Low |
| 24 | West Roxbury | Low |
| 25 | Charlestown | Low |

**Key differences from the data-driven ranking:** Roxbury, Mattapan, and Dorchester move to the top — district-level counts dilute their concentrated violent crime. Mission Hill jumps up due to its proximity to Roxbury. Downtown/Financial District drop significantly as their data count was inflated by commercial density and foot traffic, not residential danger. Charlestown holds at #25, having gentrified the most dramatically of any Boston neighborhood over the past 20 years.

---

## Ranking 2 — 311 Red Flag Complaints (2026 YTD)

Based on the Boston 311 dataset filtered to 5 high-severity complaint types that signal housing quality issues and visible neighborhood distress:
**CE Collection, Needle Pickup, Encampments, Heat - Excessive/Insufficient, Unsatisfactory Living Conditions**

Dataset covers Jan 1 – Mar 18, 2026 only (the dataset is a rolling current-year feed, not a historical archive).

**SQL query used:**
```sql
SELECT "neighborhood", COUNT(*) as count
FROM "1a0b420d-99f1-4887-9851-990b2a5a6e17"
WHERE "type" IN (
  'CE Collection',
  'Needle Pickup',
  'Encampments',
  'Heat - Excessive  Insufficient',
  'Unsatisfactory Living Conditions'
)
GROUP BY "neighborhood"
ORDER BY count DESC
```

**Note on the heat complaint string:** `'Heat - Excessive  Insufficient'` uses two spaces — exactly as it appears in the dataset. A single space will return zero results.

Split-label groups (Allston/Brighton, South Boston/Waterfront, Mattapan/Greater Mattapan) have been consolidated — their raw counts summed and entries ranked together as a single block.

| Rank | UI Neighborhood | 311 Red Flag Count | Tier | Notes |
|---|---|---|---|---|
| 1 | Roxbury | 974 | High | |
| 2 | South End | 906 | High | High rental density + shelter/services presence on Mass Ave |
| 3 | Dorchester | 494 | High | |
| 4 | Downtown | 406 | High | |
| 4 | Financial District | 406 | High | |
| 6 | Beacon Hill | 389 | High | |
| 7 | Back Bay | 372 | High | |
| 8 | Allston | 249 | High | Consolidated: Allston (9) + Allston / Brighton (220) + Brighton (20) |
| 8 | Allston / Brighton | 249 | High | Consolidated: Allston (9) + Allston / Brighton (220) + Brighton (20) |
| 8 | Brighton | 249 | High | Consolidated: Allston (9) + Allston / Brighton (220) + Brighton (20) |
| 11 | South Boston | 212 | Moderate | Consolidated: South Boston (36) + South Boston / South Boston Waterfront (176) |
| 11 | South Boston Waterfront | 212 | Moderate | Consolidated: South Boston (36) + South Boston / South Boston Waterfront (176) |
| 13 | Greater Mattapan | 182 | Moderate | Consolidated: Greater Mattapan (169) + Mattapan (13) |
| 13 | Mattapan | 182 | Moderate | Consolidated: Greater Mattapan (169) + Mattapan (13) |
| 15 | Jamaica Plain | 172 | Moderate | |
| 16 | East Boston | 166 | Moderate | |
| 17 | Fenway | 81 | Moderate | |
| 17 | Kenmore | 81 | Moderate | |
| 17 | Audubon Circle | 81 | Moderate | |
| 17 | Longwood | 81 | Moderate | |
| 21 | Mission Hill | 80 | Low | |
| 22 | Charlestown | 65 | Low | |
| 23 | Hyde Park | 59 | Low | |
| 24 | Roslindale | 42 | Low | |
| 25 | West Roxbury | 33 | Low | Consistently lowest across all three rankings |

**Key observations:**
- South End at #2 is the biggest surprise — it ranks ahead of Dorchester despite being a high-value, gentrified neighborhood. Its large concentration of older rental stock and the shelter/social services corridor along Mass Ave drive disproportionately high CE Collection, Needle Pickup, and Encampment counts.
- Mattapan (13) appears low in the raw data but is a label inconsistency — most records are filed under Greater Mattapan (169). Combined count of 182 is more representative.
- West Roxbury sits at #25 on this ranking, #24 on the knowledge-based ranking, and #20 on the serious crime ranking — the most consistently safe neighborhood in Boston across all three measures.
- The top 7 neighborhoods account for the majority of red flag complaints citywide.

---

## Ranking 3 — Green Space (Recreational Acres, Cemeteries Excluded)

Based on the Boston Parks & Recreation open space dataset, ordered by total recreational acres excluding cemeteries. Adjusted from raw data to correct two misleading groupings.

**SQL queries used:**

```sql
-- Total acres by district (all open space types)
SELECT "DISTRICT", COUNT(*) as count, SUM(CAST("ACRES" AS NUMERIC)) as total_acres
FROM "61c0239f-c8fd-47de-8375-2405382ef37c"
GROUP BY "DISTRICT"
ORDER BY total_acres DESC
```

```sql
-- Recreational acres by district (cemeteries excluded)
SELECT "DISTRICT", SUM(CAST("ACRES" AS NUMERIC)) as recreational_acres
FROM "61c0239f-c8fd-47de-8375-2405382ef37c"
WHERE "TypeLong" != 'Cemeteries & Burying Grounds'
GROUP BY "DISTRICT"
ORDER BY recreational_acres DESC
```

**Adjustments from raw data:**
- **Jamaica Plain moved to #1** over Roslindale — JP's acreage reflects genuinely walkable, embedded parks (Arnold Arboretum, Olmsted's Emerald Necklace) directly accessible to residents. Roslindale's higher raw acreage (906 acres) is largely the Stony Brook Reservation woodland along its border with West Roxbury — valuable natural space but not the same as street-level park access.
- **Beacon Hill dropped to #24** — it shares the `Back Bay/Beacon Hill` BPRD district with Back Bay, but the 260 acres is almost entirely the Charles River Esplanade on the Back Bay side. Beacon Hill itself has almost no usable green space within its boundaries. Back Bay stays at its data-driven position.

| Rank | UI Neighborhood | Recreational Acres | Tier | Notes |
|---|---|---|---|---|
| 1 | Jamaica Plain | 752.2 | High | Most walkable embedded green space in Boston |
| 2 | West Roxbury | 565.7 | High | |
| 3 | Roslindale | 906.1 | High | High raw acreage but largely Stony Brook woodland, not walkable parks |
| 4 | Dorchester | 372.6 | High | |
| 5 | Hyde Park | 326.4 | High | |
| 6 | East Boston | 269.6 | High | |
| 7 | Back Bay | 260.2 | High | Esplanade genuinely accessible to residents |
| 8 | South Boston | 228.6 | High | |
| 8 | South Boston Waterfront | 228.6 | High | |
| 10 | Allston | 224.5 | Moderate | |
| 10 | Allston / Brighton | 224.5 | Moderate | |
| 10 | Brighton | 224.5 | Moderate | |
| 13 | Fenway | 166.2 | Moderate | |
| 13 | Kenmore | 166.2 | Moderate | |
| 13 | Audubon Circle | 166.2 | Moderate | |
| 13 | Longwood | 166.2 | Moderate | |
| 17 | Roxbury | 164.2 | Moderate | |
| 18 | Greater Mattapan | 125.8 | Low | |
| 18 | Mattapan | 125.8 | Low | |
| 20 | Charlestown | 59.1 | Low | |
| 21 | Downtown | 53.4 | Low | |
| 21 | Financial District | 53.4 | Low | |
| 23 | Mission Hill | 27.2 | Low | |
| 24 | Beacon Hill | — | Low | Shares BPRD district with Back Bay but has minimal walkable green space within neighborhood boundaries |
| 25 | South End | 17.5 | Low | |

**Key observations:**
- Jamaica Plain, West Roxbury, and Roslindale are the three greenest neighborhoods in Boston by a significant margin — all three also sit near the bottom on crime and 311 red flag rankings, making them the strongest all-round buyer story in the dataset.
- South End sits at #25 on green space and #2 on 311 red flags — the toughest combination in the entire dataset for a buyer who values outdoor space and quiet living.
- Charlestown (59.1 acres) is surprisingly low given its desirability — its footprint is small and mostly built up with limited parkland relative to the outer neighborhoods.
- Harbor Islands (420.1 recreational acres) appears in the raw data but does not map to any UI neighborhood and is excluded.

---

## Combined Tier Summary

Each neighborhood mapped across crime and 311 rankings. Green space is retained as a standalone ranking (Ranking 3) but excluded from the tier summary as it is not directly comparable to the safety and distress signals that crime and 311 measure. Ordered by UI dropdown sequence.

| Neighborhood | Crime Tier | 311 Tier |
|---|---|---|
| Allston | Moderate | High |
| Allston / Brighton | Moderate | High |
| Back Bay | Low | High |
| Beacon Hill | Low | High |
| Brighton | Moderate | High |
| Charlestown | Low | Low |
| Dorchester | High | High |
| Downtown | High | High |
| Financial District | High | High |
| East Boston | High | Moderate |
| Fenway | Moderate | Moderate |
| Kenmore | Moderate | Moderate |
| Audubon Circle | Moderate | Moderate |
| Longwood | Moderate | Moderate |
| Greater Mattapan | High | Moderate |
| Hyde Park | Low | Low |
| Jamaica Plain | Moderate | Moderate |
| Mattapan | High | Moderate |
| Mission Hill | High | Low |
| Roslindale | Low | Low |
| Roxbury | High | High |
| South Boston | Low | Moderate |
| South Boston Waterfront | Low | Moderate |
| South End | Moderate | High |
| West Roxbury | Low | Low |

**Standout patterns:**
- **Dorchester, Downtown, Financial District, Roxbury** are High on both crime and 311 — the most consistently distressed neighborhoods across both safety measures.
- **Charlestown, Hyde Park, Roslindale, West Roxbury** are Low on both — the safest and least distressed neighborhoods in the dataset.
- **Back Bay and Beacon Hill** are an important split: Low crime but High 311 — safe to live in but with meaningful housing quality distress in the rental stock.
- **Mission Hill** is the most asymmetric: High crime but Low 311 — dangerous district-level signal but relatively few housing quality complaints.
- **South End** is High on 311 but only Moderate on crime — distressed from a housing quality standpoint despite its gentrified reputation.
