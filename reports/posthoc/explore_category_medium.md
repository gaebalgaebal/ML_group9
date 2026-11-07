# Where new-item exploration happens, by department and product group (medium)

**post-hoc, development data, descriptive, not confirmatory.** For every user and every basket from the 2nd basket on (eval split: history baskets, then the target basket), an item is a new-item event when it is not in any earlier basket of the user, otherwise a repeat event (distinct items per basket). Per category: volume = new-item events and their share of all new-item events; propensity = new-item rate = new-item events / all events in the category; lift = share of new-item events / share of all events; breadth = share of users with >= 1 new-item event in the category (over all users and over users who bought anything in the category in any basket of the split); known-category share = share of the new-item events whose category was already in the user's earlier baskets. Predictability (target basket only): hit rate of the E2 user-CF top 10 and of flat popularity excluding history on the target new items of the category, the share of E2 slots in the category next to the share of target new items. CIs are user-level bootstrap percentiles. Rankings and figures describe the development data only; no causal reading is implied.

- Code commit `9c7e6f90b7ae29e817eb65d18e5c01e1530057c3`, 2026-11-07T07:31:14.680701+00:00; profile `medium`; bootstrap n = 1000, seed 42.
- Category levels are the pipeline's L1 / L2 columns from items.parquet (configs/common.yaml levels): Instacart department / aisle (identical to the processed coarse_category / fine_category columns), Complete Journey DEPARTMENT / commodity (the processed Complete Journey frame columns hold commodity / sub-commodity, which are not used here). Names are reported as stored; 'missing' labels are flagged.
- Support filter for fine-level rankings: >= 200 new-item events (Instacart), >= 100 (Complete Journey) in the event set ranked; coarse level unfiltered.
- Inputs (sha256 in the JSON): instacart: 5 files, complete_journey: 5 files.
- Runtime 197.0 s; peak working set 3469 MiB (in-process).

## User-CF selections

| dataset | k | weighting | source | equals the expected medium selection |
| --- | --- | --- | --- | --- |
| Instacart | 100 | binary | `C:\dev_ML-wt-w8\reports\metrics\medium_instacart_tuning.json` (commit `6936873dd9b541a3a177643e97700182b8d729d8`) | yes |
| Complete Journey | 20 | binary | `C:\dev_ML-wt-w8\reports\metrics\medium_complete_journey_tuning.json` (commit `6936873dd9b541a3a177643e97700182b8d729d8`) | yes |

## Instacart

- Eval split: 20,000 users; 3,355,343 distinct (user, basket, item) rows; 3,154,596 events from the 2nd basket on, 1,182,033 new-item events (overall rate 0.375; history 0.373, target 0.403); target events 211,360 of which new 85,172.
- Levels: coarse = department (coarse_category) (21 categories, 1 named 'missing'); fine = aisle (fine_category) (134 categories, 1 named 'missing'). Fine-level support filter >= 200 new-item events: passing all 134, history 134, target 84; target new items >= 200: 84.
- E2 user-CF k = 100, weighting binary (verified against the reference tuning document); overall item-level new-item hit rate of the top 10: E2 0.054, flat popularity 0.039 (85,172 target new items; 200,000 E2 slots).

### Instacart: coarse categories, eval split, history + target (sorted by volume)

| category | events | new-item events | share of new events | rate [95 % CI] | lift [95 % CI] | breadth all / buyers | new item in a known category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| produce | 932,091 | 289,628 | 24.5 % | 0.311 [0.307, 0.314] | 0.83 [0.82, 0.83] | 92.4 % / 97.6 % | 96.8 % |
| dairy eggs | 529,107 | 153,040 | 12.9 % | 0.289 [0.285, 0.293] | 0.77 [0.77, 0.78] | 89.8 % / 96.3 % | 94.1 % |
| pantry | 181,639 | 115,112 | 9.7 % | 0.634 [0.628, 0.640] | 1.69 [1.68, 1.71] | 82.6 % / 95.9 % | 86.9 % |
| snacks | 275,705 | 109,438 | 9.3 % | 0.397 [0.391, 0.403] | 1.06 [1.05, 1.07] | 82.3 % / 95.2 % | 87.3 % |
| frozen | 215,723 | 91,028 | 7.7 % | 0.422 [0.417, 0.428] | 1.13 [1.11, 1.14] | 76.5 % / 94.5 % | 85.9 % |
| beverages | 261,750 | 80,545 | 6.8 % | 0.308 [0.302, 0.314] | 0.82 [0.81, 0.83] | 79.9 % / 93.1 % | 85.4 % |
| canned goods | 103,060 | 53,113 | 4.5 % | 0.515 [0.508, 0.523] | 1.38 [1.36, 1.39] | 63.1 % / 93.8 % | 75.7 % |
| dry goods pasta | 84,673 | 42,367 | 3.6 % | 0.500 [0.492, 0.509] | 1.34 [1.32, 1.35] | 59.1 % / 93.6 % | 72.4 % |
| household | 71,023 | 40,862 | 3.5 % | 0.575 [0.566, 0.584] | 1.54 [1.51, 1.56] | 53.6 % / 94.6 % | 73.1 % |
| bakery | 113,418 | 37,945 | 3.2 % | 0.335 [0.328, 0.341] | 0.89 [0.88, 0.91] | 64.1 % / 91.7 % | 73.5 % |
| deli | 101,724 | 36,108 | 3.1 % | 0.355 [0.348, 0.362] | 0.95 [0.93, 0.96] | 61.6 % / 91.3 % | 71.2 % |
| personal care | 43,248 | 28,679 | 2.4 % | 0.663 [0.653, 0.673] | 1.77 [1.74, 1.80] | 45.0 % / 93.5 % | 65.7 % |
| breakfast | 66,077 | 27,453 | 2.3 % | 0.415 [0.406, 0.425] | 1.11 [1.08, 1.13] | 52.6 % / 91.2 % | 63.2 % |
| meat seafood | 67,642 | 26,846 | 2.3 % | 0.397 [0.389, 0.404] | 1.06 [1.04, 1.08] | 52.2 % / 91.6 % | 66.8 % |
| babies | 43,966 | 17,332 | 1.5 % | 0.394 [0.379, 0.410] | 1.05 [1.01, 1.09] | 16.4 % / 93.3 % | 71.9 % |
| international | 25,748 | 15,663 | 1.3 % | 0.608 [0.597, 0.620] | 1.62 [1.60, 1.65] | 36.4 % / 91.7 % | 51.2 % |
| alcohol | 13,629 | 5,302 | 0.4 % | 0.389 [0.364, 0.416] | 1.04 [0.97, 1.11] | 6.9 % / 86.3 % | 72.8 % |
| missing (missing label) | 8,083 | 4,698 | 0.4 % | 0.581 [0.560, 0.600] | 1.55 [1.50, 1.60] | 15.4 % / 96.4 % | 28.8 % |
| pets | 9,170 | 3,418 | 0.3 % | 0.373 [0.348, 0.398] | 0.99 [0.93, 1.06] | 7.0 % / 89.3 % | 52.3 % |
| other | 3,611 | 2,104 | 0.2 % | 0.583 [0.551, 0.616] | 1.56 [1.47, 1.64] | 8.8 % / 91.5 % | 15.3 % |
| bulk | 3,509 | 1,352 | 0.1 % | 0.385 [0.357, 0.416] | 1.03 [0.95, 1.11] | 5.9 % / 88.4 % | 14.7 % |

### Instacart: coarse rankings, eval history baskets (21 of 21 categories eligible)

| # | volume (new events; share) | propensity (rate [CI]; new events) | lift top (lift [CI]) | lift bottom (lift [CI]) | breadth (all users / buyers) |
| --- | --- | --- | --- | --- | --- |
| 1 | produce (268,656; 24.5 %) | personal care (0.663 [0.653, 0.674]; n 26,547) | personal care (1.78 [1.75, 1.81]) | dairy eggs (0.77 [0.76, 0.78]) | produce (90.6 % / 95.8 %) |
| 2 | dairy eggs (142,241; 13.0 %) | pantry (0.633 [0.627, 0.640]; n 107,182) | pantry (1.70 [1.68, 1.72]) | beverages (0.82 [0.81, 0.83]) | dairy eggs (87.4 % / 93.7 %) |
| 3 | pantry (107,182; 9.8 %) | international (0.608 [0.596, 0.621]; n 14,527) | international (1.63 [1.60, 1.66]) | produce (0.83 [0.82, 0.84]) | pantry (79.2 % / 92.0 %) |
| 4 | snacks (101,838; 9.3 %) | other (0.580 [0.545, 0.614]; n 1,934) | other (1.56 [1.46, 1.65]) | bakery (0.89 [0.88, 0.91]) | snacks (79.0 % / 91.3 %) |
| 5 | frozen (84,362; 7.7 %) | missing (missing label) (0.577 [0.553, 0.598]; n 3,908) | missing (missing label) (1.55 [1.49, 1.61]) | deli (0.95 [0.93, 0.96]) | beverages (76.3 % / 89.0 %) |
| 6 | beverages (74,640; 6.8 %) | household (0.575 [0.565, 0.584]; n 37,636) | household (1.54 [1.52, 1.57]) | pets (0.99 [0.91, 1.06]) | frozen (73.3 % / 90.7 %) |
| 7 | canned goods (49,293; 4.5 %) | canned goods (0.514 [0.507, 0.522]; n 49,293) | canned goods (1.38 [1.36, 1.40]) | bulk (1.03 [0.96, 1.12]) | bakery (60.9 % / 87.1 %) |
| 8 | dry goods pasta (39,357; 3.6 %) | dry goods pasta (0.500 [0.491, 0.509]; n 39,357) | dry goods pasta (1.34 [1.32, 1.36]) | alcohol (1.04 [0.97, 1.11]) | canned goods (59.9 % / 89.0 %) |
| 9 | household (37,636; 3.4 %) | frozen (0.421 [0.415, 0.427]; n 84,362) | frozen (1.13 [1.12, 1.14]) | babies (1.05 [1.01, 1.09]) | deli (58.3 % / 86.4 %) |
| 10 | bakery (35,236; 3.2 %) | breakfast (0.414 [0.403, 0.423]; n 25,477) | breakfast (1.11 [1.08, 1.13]) | snacks (1.06 [1.05, 1.07]) | dry goods pasta (55.9 % / 88.6 %) |

### Instacart: coarse rankings, eval target baskets (21 of 21 categories eligible)

| # | volume (new events; share) | propensity (rate [CI]; new events) | lift top (lift [CI]) | lift bottom (lift [CI]) | breadth (all users / buyers) |
| --- | --- | --- | --- | --- | --- |
| 1 | produce (20,972; 24.6 %) | personal care (0.659 [0.641, 0.679]; n 2,132) | personal care (1.64 [1.59, 1.69]) | dairy eggs (0.81 [0.80, 0.82]) | produce (44.6 % / 47.1 %) |
| 2 | dairy eggs (10,799; 12.7 %) | pantry (0.639 [0.628, 0.649]; n 7,930) | pantry (1.59 [1.56, 1.61]) | produce (0.83 [0.82, 0.84]) | dairy eggs (32.6 % / 34.9 %) |
| 3 | pantry (7,930; 9.3 %) | other (0.618 [0.561, 0.672]; n 170) | other (1.53 [1.39, 1.67]) | beverages (0.85 [0.83, 0.87]) | pantry (24.6 % / 28.6 %) |
| 4 | snacks (7,600; 8.9 %) | international (0.609 [0.587, 0.634]; n 1,136) | international (1.51 [1.46, 1.57]) | bakery (0.93 [0.90, 0.95]) | snacks (23.0 % / 26.6 %) |
| 5 | frozen (6,666; 7.8 %) | missing (missing label) (0.605 [0.572, 0.634]; n 790) | missing (missing label) (1.50 [1.42, 1.58]) | deli (0.95 [0.92, 0.98]) | frozen (20.7 % / 25.5 %) |
| 6 | beverages (5,905; 6.9 %) | household (0.584 [0.569, 0.599]; n 3,226) | household (1.45 [1.41, 1.49]) | bulk (0.98 [0.81, 1.16]) | beverages (19.7 % / 23.0 %) |
| 7 | canned goods (3,820; 4.5 %) | canned goods (0.528 [0.513, 0.542]; n 3,820) | canned goods (1.31 [1.28, 1.35]) | alcohol (1.00 [0.89, 1.10]) | canned goods (12.8 % / 19.1 %) |
| 8 | household (3,226; 3.8 %) | dry goods pasta (0.507 [0.491, 0.522]; n 3,010) | dry goods pasta (1.26 [1.22, 1.29]) | meat seafood (1.04 [1.00, 1.07]) | bakery (11.6 % / 16.5 %) |
| 9 | dry goods pasta (3,010; 3.5 %) | babies (0.452 [0.416, 0.489]; n 1,122) | babies (1.12 [1.03, 1.21]) | snacks (1.05 [1.03, 1.07]) | dry goods pasta (11.0 % / 17.4 %) |
| 10 | bakery (2,709; 3.2 %) | breakfast (0.442 [0.424, 0.458]; n 1,976) | breakfast (1.10 [1.06, 1.14]) | pets (1.06 [0.92, 1.22]) | household (10.9 % / 19.2 %) |

### Instacart: fine rankings, eval history baskets (134 of 134 categories eligible)

| # | volume (new events; share) | propensity (rate [CI]; new events) | lift top (lift [CI]) | lift bottom (lift [CI]) | breadth (all users / buyers) |
| --- | --- | --- | --- | --- | --- |
| 1 | fresh vegetables (115,488; 10.5 %) | spices seasonings (0.842 [0.832, 0.851]; n 16,079) | spices seasonings (2.26 [2.23, 2.29]) | milk (0.46 [0.45, 0.47]) | fresh fruits (79.3 % / 90.7 %) |
| 2 | fresh fruits (79,574; 7.3 %) | beauty (0.819 [0.777, 0.860]; n 443) | beauty (2.20 [2.08, 2.31]) | water seltzer sparkling water (0.61 [0.59, 0.63]) | fresh vegetables (73.2 % / 92.9 %) |
| 3 | packaged vegetables fruits (51,765; 4.7 %) | first aid (0.811 [0.762, 0.859]; n 770) | first aid (2.17 [2.04, 2.31]) | fresh fruits (0.63 [0.63, 0.64]) | packaged vegetables fruits (70.3 % / 88.7 %) |
| 4 | yogurt (35,853; 3.3 %) | baking supplies decor (0.805 [0.752, 0.848]; n 1,768) | baking supplies decor (2.16 [2.02, 2.28]) | eggs (0.66 [0.64, 0.67]) | packaged cheese (58.2 % / 86.9 %) |
| 5 | packaged cheese (33,525; 3.1 %) | eye ear care (0.795 [0.755, 0.833]; n 557) | eye ear care (2.13 [2.02, 2.24]) | packaged produce (0.71 [0.67, 0.75]) | yogurt (52.7 % / 84.5 %) |
| 6 | chips pretzels (24,311; 2.2 %) | more household (0.764 [0.732, 0.795]; n 1,187) | more household (2.05 [1.96, 2.14]) | cream (0.72 [0.70, 0.75]) | chips pretzels (47.8 % / 85.1 %) |
| 7 | ice cream ice (21,453; 2.0 %) | deodorants (0.753 [0.712, 0.789]; n 1,007) | deodorants (2.02 [1.91, 2.12]) | yogurt (0.72 [0.71, 0.74]) | frozen produce (43.1 % / 85.0 %) |
| 8 | baking ingredients (20,204; 1.8 %) | shave needs (0.752 [0.701, 0.793]; n 643) | shave needs (2.02 [1.88, 2.13]) | soy lactosefree (0.73 [0.71, 0.74]) | bread (42.3 % / 81.1 %) |
| 9 | frozen produce (19,777; 1.8 %) | cold flu allergy (0.746 [0.714, 0.778]; n 1,800) | cold flu allergy (2.00 [1.91, 2.09]) | bread (0.77 [0.75, 0.79]) | milk (41.8 % / 76.4 %) |
| 10 | water seltzer sparkling water (17,347; 1.6 %) | skin care (0.742 [0.689, 0.789]; n 685) | skin care (1.99 [1.84, 2.12]) | refrigerated (0.77 [0.75, 0.80]) | water seltzer sparkling water (40.3 % / 80.2 %) |

### Instacart: fine rankings, eval target baskets (84 of 134 categories eligible)

| # | volume (new events; share) | propensity (rate [CI]; new events) | lift top (lift [CI]) | lift bottom (lift [CI]) | breadth (all users / buyers) |
| --- | --- | --- | --- | --- | --- |
| 1 | fresh vegetables (9,050; 10.6 %) | spices seasonings (0.829 [0.808, 0.848]; n 1,204) | spices seasonings (2.06 [2.00, 2.11]) | milk (0.51 [0.48, 0.53]) | fresh vegetables (25.3 % / 32.1 %) |
| 2 | fresh fruits (6,081; 7.1 %) | food storage (0.762 [0.721, 0.804]; n 337) | food storage (1.89 [1.79, 2.00]) | fresh fruits (0.65 [0.64, 0.67]) | fresh fruits (22.0 % / 25.1 %) |
| 3 | packaged vegetables fruits (4,182; 4.9 %) | marinades meat preparation (0.733 [0.688, 0.777]; n 302) | marinades meat preparation (1.82 [1.71, 1.93]) | water seltzer sparkling water (0.67 [0.64, 0.71]) | packaged vegetables fruits (16.6 % / 21.0 %) |
| 4 | yogurt (2,655; 3.1 %) | baking ingredients (0.696 [0.671, 0.719]; n 1,348) | baking ingredients (1.73 [1.67, 1.78]) | packaged produce (0.68 [0.63, 0.73]) | packaged cheese (11.0 % / 16.4 %) |
| 5 | packaged cheese (2,632; 3.1 %) | cleaning products (0.693 [0.657, 0.724]; n 633) | cleaning products (1.72 [1.63, 1.80]) | eggs (0.68 [0.64, 0.72]) | yogurt (9.2 % / 14.7 %) |
| 6 | chips pretzels (1,855; 2.2 %) | condiments (0.662 [0.636, 0.687]; n 991) | condiments (1.64 [1.58, 1.71]) | cream (0.75 [0.70, 0.80]) | chips pretzels (7.5 % / 13.3 %) |
| 7 | ice cream ice (1,703; 2.0 %) | soap (0.650 [0.599, 0.698]; n 249) | soap (1.61 [1.48, 1.73]) | soy lactosefree (0.77 [0.73, 0.81]) | ice cream ice (6.2 % / 13.6 %) |
| 8 | frozen produce (1,539; 1.8 %) | salad dressing toppings (0.646 [0.609, 0.681]; n 465) | salad dressing toppings (1.60 [1.51, 1.69]) | yogurt (0.79 [0.76, 0.82]) | frozen produce (6.2 % / 12.2 %) |
| 9 | water seltzer sparkling water (1,487; 1.7 %) | asian foods (0.643 [0.615, 0.673]; n 704) | asian foods (1.60 [1.53, 1.67]) | refrigerated (0.81 [0.76, 0.86]) | water seltzer sparkling water (6.0 % / 12.0 %) |
| 10 | soup broth bouillon (1,361; 1.6 %) | honeys syrups nectars (0.639 [0.591, 0.686]; n 273) | honeys syrups nectars (1.59 [1.47, 1.70]) | bread (0.81 [0.77, 0.85]) | soy lactosefree (5.6 % / 11.3 %) |

### Instacart: predictability of target new items, coarse level (eligible categories by target volume)

| category | target new items | share of target new items | E2 hit rate [95 % CI] | popularity hit rate | E2 slot share | slot share - target share | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| produce | 20,972 | 24.6 % | 0.203 [0.198, 0.208] | 0.154 | 88.2 % | +63.6 pp | 3.58 |
| dairy eggs | 10,799 | 12.7 % | 0.007 [0.006, 0.009] | 0.010 | 2.8 % | -9.9 pp | 0.22 |
| pantry | 7,930 | 9.3 % | 0.003 [0.002, 0.004] | 0.000 | 1.0 % | -8.3 pp | 0.10 |
| snacks | 7,600 | 8.9 % | 0.006 [0.004, 0.008] | 0.000 | 1.6 % | -7.3 pp | 0.18 |
| frozen | 6,666 | 7.8 % | 0.005 [0.004, 0.007] | 0.000 | 1.2 % | -6.6 pp | 0.16 |
| beverages | 5,905 | 6.9 % | 0.012 [0.009, 0.015] | 0.000 | 2.6 % | -4.3 pp | 0.38 |
| canned goods | 3,820 | 4.5 % | 0.002 [0.001, 0.003] | 0.000 | 0.3 % | -4.2 pp | 0.06 |
| household | 3,226 | 3.8 % | 0.002 [0.001, 0.003] | 0.000 | 0.1 % | -3.6 pp | 0.04 |
| dry goods pasta | 3,010 | 3.5 % | 0.001 [0.000, 0.002] | 0.000 | 0.0 % | -3.5 pp | 0.01 |
| bakery | 2,709 | 3.2 % | 0.007 [0.004, 0.010] | 0.000 | 0.7 % | -2.5 pp | 0.21 |
| deli | 2,558 | 3.0 % | 0.004 [0.002, 0.007] | 0.000 | 0.4 % | -2.6 pp | 0.13 |
| personal care | 2,132 | 2.5 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -2.5 pp | 0.00 |
| breakfast | 1,976 | 2.3 % | 0.002 [0.000, 0.004] | 0.000 | 0.3 % | -2.1 pp | 0.11 |
| meat seafood | 1,909 | 2.2 % | 0.012 [0.007, 0.017] | 0.000 | 0.4 % | -1.8 pp | 0.19 |
| international | 1,136 | 1.3 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.3 pp | 0.01 |
| babies | 1,122 | 1.3 % | 0.001 [0.000, 0.003] | 0.000 | 0.0 % | -1.3 pp | 0.03 |
| missing (missing label) | 790 | 0.9 % | 0.001 [0.000, 0.004] | 0.000 | 0.0 % | -0.9 pp | 0.01 |
| alcohol | 350 | 0.4 % | 0.017 [0.006, 0.032] | 0.000 | 0.3 % | -0.1 pp | 0.72 |
| pets | 304 | 0.4 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.4 pp | 0.00 |
| other | 170 | 0.2 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.2 pp | 0.00 |
| bulk | 88 | 0.1 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.1 pp | 0.00 |

Mismatch (21 categories considered; 1 with an E2 slot share above their target new-item share, 20 below): the largest differences in both directions (rows past the counts are the least negative / least positive):

| # | over-served (slot share vs target share) | under-served (slot share vs target share) |
| --- | --- | --- |
| 1 | produce (88.2 % vs 24.6 %; +63.6 pp) | dairy eggs (2.8 % vs 12.7 %; -9.9 pp) |
| 2 | bulk (0.0 % vs 0.1 %; -0.1 pp) | pantry (1.0 % vs 9.3 %; -8.3 pp) |
| 3 | alcohol (0.3 % vs 0.4 %; -0.1 pp) | snacks (1.6 % vs 8.9 %; -7.3 pp) |
| 4 | other (0.0 % vs 0.2 %; -0.2 pp) | frozen (1.2 % vs 7.8 %; -6.6 pp) |
| 5 | pets (0.0 % vs 0.4 %; -0.4 pp) | beverages (2.6 % vs 6.9 %; -4.3 pp) |
| 6 | missing (missing label) (0.0 % vs 0.9 %; -0.9 pp) | canned goods (0.3 % vs 4.5 %; -4.2 pp) |
| 7 | babies (0.0 % vs 1.3 %; -1.3 pp) | household (0.1 % vs 3.8 %; -3.6 pp) |
| 8 | international (0.0 % vs 1.3 %; -1.3 pp) | dry goods pasta (0.0 % vs 3.5 %; -3.5 pp) |
| 9 | meat seafood (0.4 % vs 2.2 %; -1.8 pp) | deli (0.4 % vs 3.0 %; -2.6 pp) |
| 10 | breakfast (0.3 % vs 2.3 %; -2.1 pp) | personal care (0.0 % vs 2.5 %; -2.5 pp) |

### Instacart: predictability of target new items, fine level (eligible categories by target volume, top 15)

| category | target new items | share of target new items | E2 hit rate [95 % CI] | popularity hit rate | E2 slot share | slot share - target share | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fresh vegetables | 9,050 | 10.6 % | 0.131 [0.125, 0.138] | 0.041 | 25.7 % | +15.0 pp | 2.42 |
| fresh fruits | 6,081 | 7.1 % | 0.344 [0.333, 0.356] | 0.375 | 40.1 % | +33.0 pp | 5.62 |
| packaged vegetables fruits | 4,182 | 4.9 % | 0.162 [0.151, 0.174] | 0.135 | 14.2 % | +9.2 pp | 2.88 |
| yogurt | 2,655 | 3.1 % | 0.006 [0.003, 0.010] | 0.000 | 0.5 % | -2.6 pp | 0.15 |
| packaged cheese | 2,632 | 3.1 % | 0.007 [0.004, 0.011] | 0.000 | 0.5 % | -2.6 pp | 0.16 |
| chips pretzels | 1,855 | 2.2 % | 0.008 [0.004, 0.012] | 0.000 | 0.4 % | -1.8 pp | 0.18 |
| ice cream ice | 1,703 | 2.0 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -2.0 pp | 0.02 |
| frozen produce | 1,539 | 1.8 % | 0.023 [0.016, 0.032] | 0.000 | 1.2 % | -0.6 pp | 0.64 |
| water seltzer sparkling water | 1,487 | 1.7 % | 0.033 [0.023, 0.043] | 0.000 | 1.5 % | -0.3 pp | 0.84 |
| soup broth bouillon | 1,361 | 1.6 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.6 pp | 0.02 |
| baking ingredients | 1,348 | 1.6 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.5 pp | 0.02 |
| soy lactosefree | 1,225 | 1.4 % | 0.000 [0.000, 0.000] | 0.000 | 0.1 % | -1.3 pp | 0.08 |
| crackers | 1,221 | 1.4 % | 0.002 [0.000, 0.006] | 0.000 | 0.1 % | -1.3 pp | 0.10 |
| spices seasonings | 1,204 | 1.4 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.4 pp | 0.00 |
| bread | 1,154 | 1.4 % | 0.012 [0.006, 0.019] | 0.000 | 0.5 % | -0.9 pp | 0.36 |

Mismatch (84 categories considered; 6 with an E2 slot share above their target new-item share, 78 below): the largest differences in both directions (rows past the counts are the least negative / least positive):

| # | over-served (slot share vs target share) | under-served (slot share vs target share) |
| --- | --- | --- |
| 1 | fresh fruits (40.1 % vs 7.1 %; +33.0 pp) | yogurt (0.5 % vs 3.1 %; -2.6 pp) |
| 2 | fresh vegetables (25.7 % vs 10.6 %; +15.0 pp) | packaged cheese (0.5 % vs 3.1 %; -2.6 pp) |
| 3 | packaged vegetables fruits (14.2 % vs 4.9 %; +9.2 pp) | ice cream ice (0.0 % vs 2.0 %; -2.0 pp) |
| 4 | packaged produce (4.9 % vs 0.7 %; +4.2 pp) | chips pretzels (0.4 % vs 2.2 %; -1.8 pp) |
| 5 | fresh herbs (3.4 % vs 1.3 %; +2.1 pp) | soup broth bouillon (0.0 % vs 1.6 %; -1.6 pp) |
| 6 | packaged poultry (0.4 % vs 0.4 %; +0.0 pp) | baking ingredients (0.0 % vs 1.6 %; -1.5 pp) |
| 7 | soft drinks (1.0 % vs 1.0 %; -0.0 pp) | spices seasonings (0.0 % vs 1.4 %; -1.4 pp) |
| 8 | body lotions soap (0.0 % vs 0.2 %; -0.2 pp) | soy lactosefree (0.1 % vs 1.4 %; -1.3 pp) |
| 9 | frozen meat seafood (0.0 % vs 0.3 %; -0.3 pp) | frozen meals (0.0 % vs 1.3 %; -1.3 pp) |
| 10 | granola (0.0 % vs 0.3 %; -0.3 pp) | crackers (0.1 % vs 1.4 %; -1.3 pp) |

### Instacart: 2 x 2 placement of fine categories (84 categories; median rate 0.423, median E2 hit rate 0.001)

- explored but hard to predict (rate above median, E2 hit rate below median): 26 categories: ice cream ice (0.477, 0.000, n 1,703); soup broth bouillon (0.555, 0.000, n 1,361); baking ingredients (0.676, 0.000, n 1,348); spices seasonings (0.841, 0.000, n 1,204); canned jarred vegetables (0.548, 0.000, n 1,095); dry pasta (0.506, 0.000, n 850); instant foods (0.476, 0.000, n 785); tea (0.443, 0.000, n 733); asian foods (0.625, 0.000, n 704); frozen appetizers sides (0.443, 0.000, n 639); tortillas flat bread (0.424, 0.000, n 588); grains rice dried goods (0.568, 0.000, n 510); hot cereal pancake mixes (0.473, 0.000, n 501); salad dressing toppings (0.629, 0.000, n 465); pickled goods olives (0.534, 0.000, n 447); doughs gelatins bake mixes (0.564, 0.000, n 407); preserved dips spreads (0.506, 0.000, n 400); dish detergents (0.587, 0.000, n 373); latino foods (0.586, 0.000, n 318); specialty cheeses (0.491, 0.000, n 310); marinades meat preparation (0.711, 0.000, n 302); oral hygiene (0.662, 0.000, n 296); canned meat seafood (0.485, 0.000, n 272); soap (0.685, 0.000, n 249); meat counter (0.435, 0.000, n 246); body lotions soap (0.652, 0.000, n 210)

- explored and easier to predict (both above median): 16 categories: frozen produce (0.427, 0.023, n 1,539); fresh herbs (0.445, 0.141, n 1,102); oils vinegars (0.628, 0.011, n 1,007); condiments (0.681, 0.004, n 991); nuts seeds dried fruit (0.458, 0.002, n 944); spreads (0.484, 0.009, n 927); paper goods (0.438, 0.002, n 866); canned meals beans (0.468, 0.009, n 820); missing (missing label) (0.581, 0.001, n 790); pasta sauce (0.479, 0.003, n 765); cookies cakes (0.437, 0.007, n 688); cleaning products (0.701, 0.002, n 633); laundry (0.562, 0.003, n 392); food storage (0.742, 0.003, n 337); buns rolls (0.427, 0.006, n 318); honeys syrups nectars (0.636, 0.004, n 273)

- rarely explored and hard to predict (both below median): 16 categories: soy lactosefree (0.273, 0.000, n 1,225); refrigerated (0.290, 0.001, n 1,140); frozen meals (0.399, 0.000, n 1,134); lunch meat (0.355, 0.001, n 995); juice nectars (0.368, 0.000, n 854); hot dogs bacon sausage (0.411, 0.000, n 811); coffee (0.340, 0.000, n 468); frozen breakfast (0.335, 0.000, n 468); frozen pizza (0.368, 0.000, n 466); frozen vegan vegetarian (0.419, 0.000, n 316); canned fruit applesauce (0.413, 0.000, n 272); tofu meat alternatives (0.375, 0.000, n 256); energy sports drinks (0.308, 0.000, n 248); prepared meals (0.331, 0.000, n 238); granola (0.385, 0.000, n 222); frozen meat seafood (0.422, 0.000, n 221)

- rarely explored but easier to predict (rate below, E2 above): 26 categories: fresh vegetables (0.371, 0.131, n 9,050); fresh fruits (0.238, 0.344, n 6,081); packaged vegetables fruits (0.323, 0.162, n 4,182); yogurt (0.273, 0.006, n 2,655); packaged cheese (0.378, 0.007, n 2,632); chips pretzels (0.379, 0.008, n 1,855); water seltzer sparkling water (0.231, 0.033, n 1,487); crackers (0.397, 0.002, n 1,221); bread (0.289, 0.012, n 1,154); energy granola bars (0.370, 0.004, n 1,091); cereal (0.401, 0.003, n 1,062); milk (0.173, 0.019, n 1,015); baby food formula (0.383, 0.001, n 968); fresh dips tapenades (0.355, 0.011, n 899); soft drinks (0.329, 0.023, n 876); eggs (0.247, 0.013, n 840); other creams cheeses (0.393, 0.007, n 825); candy chocolate (0.397, 0.004, n 763); butter (0.382, 0.005, n 616); cream (0.271, 0.007, n 576); packaged produce (0.265, 0.262, n 557); breakfast bakery (0.310, 0.004, n 534); fruit vegetable snacks (0.384, 0.009, n 438); popcorn jerky (0.378, 0.005, n 417); packaged poultry (0.355, 0.064, n 314); poultry counter (0.345, 0.007, n 294)

### Instacart: robustness

Spearman rank correlation of the fit-split and eval-split category values (history + target events; eligible eval categories present in both). The fit split shares most of its events with the eval split (Instacart: every basket but the last; Complete Journey: an earlier cutoff), so these correlations check the stability of the category values rather than provide independent evidence:

| level | categories | volume (new-item events) | propensity (rate) |
| --- | --- | --- | --- |
| coarse | 21 | 1.000 | 0.999 |
| fine | 134 | 1.000 | 1.000 |

Coarse new-item rate by T5 history-length bin (eval, history + target events; categories sorted by volume; '-' = no events in the bin):

| category | (0,4] (4,143 users) | (4,6] (2,933 users) | (6,10] (3,841 users) | (10,20] (4,470 users) | (20,inf) (4,613 users) |
| --- | --- | --- | --- | --- | --- |
| overall | 0.647 | 0.587 | 0.520 | 0.435 | 0.291 |
| produce | 0.595 | 0.528 | 0.458 | 0.369 | 0.232 |
| dairy eggs | 0.576 | 0.515 | 0.441 | 0.358 | 0.209 |
| snacks | 0.661 | 0.611 | 0.534 | 0.458 | 0.319 |
| pantry | 0.835 | 0.807 | 0.751 | 0.699 | 0.553 |
| frozen | 0.661 | 0.602 | 0.539 | 0.464 | 0.339 |
| beverages | 0.566 | 0.497 | 0.443 | 0.353 | 0.234 |
| canned goods | 0.759 | 0.697 | 0.649 | 0.566 | 0.424 |
| household | 0.826 | 0.753 | 0.727 | 0.630 | 0.470 |
| dry goods pasta | 0.745 | 0.691 | 0.635 | 0.545 | 0.413 |
| bakery | 0.617 | 0.539 | 0.487 | 0.405 | 0.249 |
| deli | 0.609 | 0.570 | 0.489 | 0.400 | 0.277 |
| meat seafood | 0.659 | 0.606 | 0.544 | 0.447 | 0.307 |
| breakfast | 0.683 | 0.616 | 0.549 | 0.479 | 0.330 |
| personal care | 0.861 | 0.821 | 0.781 | 0.708 | 0.583 |
| international | 0.786 | 0.752 | 0.718 | 0.674 | 0.533 |
| babies | 0.689 | 0.651 | 0.603 | 0.465 | 0.323 |
| alcohol | 0.599 | 0.560 | 0.539 | 0.433 | 0.290 |
| missing (missing label) | 0.751 | 0.705 | 0.615 | 0.643 | 0.510 |
| pets | 0.674 | 0.617 | 0.496 | 0.367 | 0.288 |
| other | 0.747 | 0.750 | 0.725 | 0.654 | 0.494 |
| bulk | 0.600 | 0.611 | 0.513 | 0.419 | 0.333 |

## Complete Journey

- Eval split: 2,069 users; 2,125,351 distinct (user, basket, item) rows; 2,097,753 events from the 2nd basket on, 1,144,106 new-item events (overall rate 0.545; history 0.546, target 0.525); target events 28,711 of which new 15,076.
- Levels: coarse = DEPARTMENT (31 categories, 0 named 'missing'); fine = commodity (302 categories, 0 named 'missing'). Fine-level support filter >= 100 new-item events: passing all 266, history 266, target 44; target new items >= 100: 44.
- E2 user-CF k = 20, weighting binary (verified against the reference tuning document); overall item-level new-item hit rate of the top 10: E2 0.015, flat popularity 0.011 (15,076 target new items; 20,690 E2 slots).

### Complete Journey: coarse categories, eval split, history + target (sorted by volume)

| category | events | new-item events | share of new events | rate [95 % CI] | lift [95 % CI] | breadth all / buyers | new item in a known category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GROCERY | 1,349,379 | 719,261 | 62.9 % | 0.533 [0.525, 0.541] | 0.98 [0.97, 0.98] | 100.0 % / 100.0 % | 99.8 % |
| DRUG GM | 227,714 | 160,916 | 14.1 % | 0.707 [0.696, 0.719] | 1.30 [1.28, 1.31] | 100.0 % / 100.0 % | 98.7 % |
| PRODUCE | 212,975 | 86,300 | 7.5 % | 0.405 [0.396, 0.415] | 0.74 [0.73, 0.76] | 99.2 % / 99.9 % | 96.9 % |
| MEAT-PCKGD | 91,569 | 53,281 | 4.7 % | 0.582 [0.572, 0.593] | 1.07 [1.05, 1.08] | 96.9 % / 99.9 % | 95.4 % |
| MEAT | 72,732 | 41,783 | 3.7 % | 0.574 [0.564, 0.585] | 1.05 [1.04, 1.07] | 95.4 % / 99.8 % | 94.8 % |
| DELI | 51,523 | 25,575 | 2.2 % | 0.496 [0.480, 0.513] | 0.91 [0.88, 0.94] | 91.9 % / 99.6 % | 90.0 % |
| PASTRY | 31,153 | 20,193 | 1.8 % | 0.648 [0.631, 0.666] | 1.19 [1.16, 1.22] | 93.0 % / 99.7 % | 90.2 % |
| NUTRITION | 26,233 | 14,326 | 1.3 % | 0.546 [0.519, 0.576] | 1.00 [0.95, 1.05] | 70.7 % / 99.0 % | 86.2 % |
| SEAFOOD-PCKGD | 8,936 | 6,166 | 0.5 % | 0.690 [0.663, 0.712] | 1.27 [1.21, 1.30] | 67.9 % / 99.2 % | 73.7 % |
| COSMETICS | 6,226 | 5,584 | 0.5 % | 0.897 [0.880, 0.915] | 1.64 [1.61, 1.68] | 53.3 % / 99.0 % | 73.3 % |
| FLORAL | 3,728 | 3,212 | 0.3 % | 0.862 [0.842, 0.882] | 1.58 [1.54, 1.62] | 48.5 % / 98.9 % | 65.1 % |
| SALAD BAR | 7,872 | 2,439 | 0.2 % | 0.310 [0.284, 0.337] | 0.57 [0.52, 0.62] | 46.7 % / 98.2 % | 57.3 % |
| SEAFOOD | 3,366 | 2,119 | 0.2 % | 0.630 [0.591, 0.670] | 1.15 [1.09, 1.23] | 33.9 % / 98.7 % | 64.6 % |
| SPIRITS | 1,700 | 993 | 0.1 % | 0.584 [0.487, 0.718] | 1.07 [0.89, 1.32] | 22.6 % / 98.5 % | 47.1 % |
| GARDEN CENTER | 651 | 578 | 0.1 % | 0.888 [0.859, 0.919] | 1.63 [1.57, 1.69] | 13.4 % / 98.2 % | 39.1 % |
| TRAVEL & LEISUR | 746 | 512 | 0.0 % | 0.686 [0.641, 0.733] | 1.26 [1.18, 1.34] | 20.7 % / 98.4 % | 15.2 % |
| CHEF SHOPPE | 597 | 412 | 0.0 % | 0.690 [0.636, 0.750] | 1.27 [1.17, 1.37] | 15.0 % / 99.7 % | 23.8 % |
| RESTAURANT | 367 | 206 | 0.0 % | 0.561 [0.476, 0.677] | 1.03 [0.87, 1.24] | 5.5 % / 100.0 % | 38.3 % |
| FROZEN GROCERY | 183 | 153 | 0.0 % | 0.836 [0.766, 0.902] | 1.53 [1.40, 1.66] | 6.5 % / 99.3 % | 11.8 % |
| DAIRY DELI | 51 | 49 | 0.0 % | 0.961 [0.904, 1.000] | 1.76 [1.65, 1.84] | 2.3 % / 100.0 % | 2.0 % |
| PHOTO | 13 | 13 | 0.0 % | 1.000 [1.000, 1.000] | 1.83 [1.81, 1.86] | 0.6 % / 100.0 % | 0.0 % |
| RX | 9 | 9 | 0.0 % | 1.000 [1.000, 1.000] | 1.83 [1.81, 1.86] | 0.4 % / 100.0 % | 0.0 % |
| DELI/SNACK BAR | 8 | 7 | 0.0 % | 0.875 [0.667, 1.000] | 1.60 [1.22, 1.85] | 0.3 % / 100.0 % | 0.0 % |
| POSTAL CENTER | 6 | 6 | 0.0 % | 1.000 [1.000, 1.000] | 1.83 [1.81, 1.86] | 0.3 % / 100.0 % | 0.0 % |
| HBC | 3 | 3 | 0.0 % | 1.000 [1.000, 1.000] | 1.83 [1.81, 1.86] | 0.1 % / 100.0 % | 0.0 % |
| TOYS | 4 | 3 | 0.0 % | 0.750 [0.500, 1.000] | 1.38 [0.91, 1.85] | 0.1 % / 100.0 % | 0.0 % |
| GRO BAKERY | 2 | 2 | 0.0 % | 1.000 [1.000, 1.000] | 1.83 [1.81, 1.86] | 0.0 % / 100.0 % | 0.0 % |
| PROD-WHS SALES | 4 | 2 | 0.0 % | 0.500 [0.333, 1.000] | 0.92 [0.60, 1.85] | 0.1 % / 100.0 % | 0.0 % |
| HOUSEWARES | 1 | 1 | 0.0 % | 1.000 [1.000, 1.000] | 1.83 [1.81, 1.86] | 0.0 % / 100.0 % | 0.0 % |
| PORK | 1 | 1 | 0.0 % | 1.000 [1.000, 1.000] | 1.83 [1.81, 1.86] | 0.0 % / 100.0 % | 0.0 % |
| VIDEO | 1 | 1 | 0.0 % | 1.000 [1.000, 1.000] | 1.83 [1.81, 1.86] | 0.0 % / 100.0 % | 0.0 % |

### Complete Journey: coarse rankings, eval history baskets (31 of 31 categories eligible)

| # | volume (new events; share) | propensity (rate [CI]; new events) | lift top (lift [CI]) | lift bottom (lift [CI]) | breadth (all users / buyers) |
| --- | --- | --- | --- | --- | --- |
| 1 | GROCERY (709,556; 62.8 %) | PHOTO (1.000 [1.000, 1.000]; n 13) | PHOTO (1.83 [1.81, 1.86]) | SALAD BAR (0.57 [0.52, 0.62]) | GROCERY (100.0 % / 100.0 %) |
| 2 | DRUG GM (158,606; 14.0 %) | RX (1.000 [1.000, 1.000]; n 8) | POSTAL CENTER (1.83 [1.81, 1.86]) | PRODUCE (0.74 [0.73, 0.76]) | DRUG GM (99.9 % / 99.9 %) |
| 3 | PRODUCE (85,413; 7.6 %) | POSTAL CENTER (1.000 [1.000, 1.000]; n 6) | HBC (1.83 [1.81, 1.86]) | DELI (0.91 [0.88, 0.94]) | PRODUCE (99.2 % / 99.9 %) |
| 4 | MEAT-PCKGD (52,626; 4.7 %) | HBC (1.000 [1.000, 1.000]; n 3) | RX (1.83 [1.81, 1.86]) | PROD-WHS SALES (0.92 [0.60, 1.85]) | MEAT-PCKGD (96.7 % / 99.7 %) |
| 5 | MEAT (41,320; 3.7 %) | GRO BAKERY (1.000 [1.000, 1.000]; n 2) | GRO BAKERY (1.83 [1.81, 1.86]) | GROCERY (0.98 [0.97, 0.98]) | MEAT (95.0 % / 99.4 %) |
| 6 | DELI (25,246; 2.2 %) | HOUSEWARES (1.000 [1.000, 1.000]; n 1) | HOUSEWARES (1.83 [1.81, 1.86]) | NUTRITION (1.00 [0.95, 1.05]) | PASTRY (92.8 % / 99.4 %) |
| 7 | PASTRY (19,943; 1.8 %) | PORK (1.000 [1.000, 1.000]; n 1) | PORK (1.83 [1.81, 1.86]) | RESTAURANT (1.02 [0.87, 1.23]) | DELI (91.4 % / 99.0 %) |
| 8 | NUTRITION (14,146; 1.3 %) | VIDEO (1.000 [1.000, 1.000]; n 1) | VIDEO (1.83 [1.81, 1.86]) | MEAT (1.05 [1.04, 1.07]) | NUTRITION (70.0 % / 98.0 %) |
| 9 | SEAFOOD-PCKGD (6,075; 0.5 %) | DAIRY DELI (0.961 [0.904, 1.000]; n 49) | DAIRY DELI (1.76 [1.65, 1.84]) | MEAT-PCKGD (1.07 [1.05, 1.08]) | SEAFOOD-PCKGD (67.4 % / 98.5 %) |
| 10 | COSMETICS (5,505; 0.5 %) | COSMETICS (0.898 [0.881, 0.915]; n 5,505) | COSMETICS (1.65 [1.61, 1.68]) | SPIRITS (1.07 [0.89, 1.32]) | COSMETICS (52.6 % / 97.8 %) |

### Complete Journey: coarse rankings, eval target baskets (20 of 20 categories eligible)

| # | volume (new events; share) | propensity (rate [CI]; new events) | lift top (lift [CI]) | lift bottom (lift [CI]) | breadth (all users / buyers) |
| --- | --- | --- | --- | --- | --- |
| 1 | GROCERY (9,705; 64.4 %) | FROZEN GROCERY (1.000 [1.000, 1.000]; n 3) | FROZEN GROCERY (1.90 [1.86, 1.95]) | TRAVEL & LEISUR (0.38 [0.00, 1.27]) | GROCERY (84.1 % / 84.1 %) |
| 2 | DRUG GM (2,310; 15.3 %) | RESTAURANT (1.000 [1.000, 1.000]; n 2) | RESTAURANT (1.90 [1.86, 1.95]) | SALAD BAR (0.49 [0.30, 0.70]) | DRUG GM (44.7 % / 44.7 %) |
| 3 | PRODUCE (887; 5.9 %) | GARDEN CENTER (1.000 [1.000, 1.000]; n 1) | GARDEN CENTER (1.90 [1.86, 1.95]) | PRODUCE (0.67 [0.63, 0.71]) | PRODUCE (24.1 % / 24.3 %) |
| 4 | MEAT-PCKGD (655; 4.3 %) | RX (1.000 [1.000, 1.000]; n 1) | RX (1.90 [1.86, 1.95]) | DELI (0.94 [0.85, 1.02]) | MEAT-PCKGD (20.1 % / 20.7 %) |
| 5 | MEAT (463; 3.1 %) | FLORAL (0.958 [0.897, 1.000]; n 46) | FLORAL (1.83 [1.70, 1.93]) | CHEF SHOPPE (0.95 [0.42, 1.49]) | MEAT (16.9 % / 17.7 %) |
| 6 | DELI (329; 2.2 %) | COSMETICS (0.840 [0.747, 0.921]; n 79) | COSMETICS (1.60 [1.42, 1.76]) | MEAT (0.96 [0.89, 1.02]) | DELI (10.7 % / 11.6 %) |
| 7 | PASTRY (250; 1.7 %) | SEAFOOD (0.814 [0.684, 0.917]; n 35) | SEAFOOD (1.55 [1.31, 1.74]) | GROCERY (0.97 [0.96, 0.98]) | PASTRY (9.6 % / 10.3 %) |
| 8 | NUTRITION (180; 1.2 %) | DRUG GM (0.728 [0.705, 0.750]; n 2,310) | DRUG GM (1.39 [1.34, 1.43]) | MEAT-PCKGD (1.02 [0.96, 1.08]) | NUTRITION (5.7 % / 7.9 %) |
| 9 | SEAFOOD-PCKGD (91; 0.6 %) | SEAFOOD-PCKGD (0.689 [0.593, 0.775]; n 91) | SEAFOOD-PCKGD (1.31 [1.13, 1.47]) | NUTRITION (1.07 [0.95, 1.19]) | SEAFOOD-PCKGD (3.5 % / 5.2 %) |
| 10 | COSMETICS (79; 0.5 %) | PASTRY (0.661 [0.608, 0.711]; n 250) | PASTRY (1.26 [1.16, 1.35]) | SPIRITS (1.09 [0.64, 1.50]) | COSMETICS (2.7 % / 5.0 %) |

### Complete Journey: fine rankings, eval history baskets (266 of 302 categories eligible)

| # | volume (new events; share) | propensity (rate [CI]; new events) | lift top (lift [CI]) | lift bottom (lift [CI]) | breadth (all users / buyers) |
| --- | --- | --- | --- | --- | --- |
| 1 | SOFT DRINKS (36,782; 3.3 %) | CONTINUITIES (0.994 [0.978, 1.000]; n 162) | CONTINUITIES (1.82 [1.79, 1.85]) | TROPICAL FRUIT (0.20 [0.19, 0.21]) | BAKED BREAD/BUNS/ROLLS (97.3 % / 99.7 %) |
| 2 | CHEESE (31,580; 2.8 %) | PERSONAL CARE APPLIANCES (0.986 [0.964, 1.000]; n 136) | PERSONAL CARE APPLIANCES (1.81 [1.76, 1.85]) | FLUID MILK PRODUCTS (0.33 [0.32, 0.33]) | SOFT DRINKS (97.2 % / 99.5 %) |
| 3 | BAG SNACKS (29,909; 2.6 %) | SPORTS MEMORABLILIA (0.984 [0.955, 1.000]; n 184) | SPORTS MEMORABLILIA (1.80 [1.75, 1.85]) | CIGARETTES (0.43 [0.39, 0.47]) | FLUID MILK PRODUCTS (97.1 % / 98.9 %) |
| 4 | FRZN MEAT/MEAT DINNERS (26,767; 2.4 %) | HALLOWEEN (0.983 [0.973, 0.991]; n 845) | HALLOWEEN (1.80 [1.77, 1.83]) | NEWSPAPER (0.47 [0.37, 0.60]) | CHEESE (96.3 % / 99.5 %) |
| 5 | BAKED BREAD/BUNS/ROLLS (26,720; 2.4 %) | SEWING (0.980 [0.947, 1.000]; n 146) | SEWING (1.80 [1.73, 1.84]) | EGGS (0.49 [0.47, 0.50]) | BAG SNACKS (95.0 % / 99.1 %) |
| 6 | BEEF (21,354; 1.9 %) | VALENTINE (0.979 [0.959, 0.996]; n 275) | VALENTINE (1.79 [1.75, 1.83]) | VEGETABLES SALAD (0.56 [0.53, 0.59]) | EGGS (91.2 % / 98.5 %) |
| 7 | SOUP (20,870; 1.8 %) | DOLLAR VALUE PRODUCTS (0.978 [0.951, 1.000]; n 131) | DOLLAR VALUE PRODUCTS (1.79 [1.74, 1.84]) | SALAD BAR (0.57 [0.52, 0.62]) | SOUP (90.5 % / 99.1 %) |
| 8 | CANDY - PACKAGED (20,686; 1.8 %) | BOOKSTORE (0.976 [0.961, 0.988]; n 1,443) | BOOKSTORE (1.79 [1.75, 1.82]) | TOMATOES (0.62 [0.60, 0.65]) | BEEF (90.4 % / 99.3 %) |
| 9 | FROZEN PIZZA (19,944; 1.8 %) | AUDIO/VIDEO PRODUCTS (0.975 [0.957, 0.989]; n 809) | AUDIO/VIDEO PRODUCTS (1.79 [1.75, 1.82]) | ADULT INCONTINENCE (0.62 [0.41, 1.00]) | CRACKERS/MISC BKD FD (90.0 % / 98.7 %) |
| 10 | VEGETABLES - SHELF STABLE (19,182; 1.7 %) | COOKWARE & BAKEWARE (0.973 [0.960, 0.984]; n 867) | COOKWARE & BAKEWARE (1.78 [1.75, 1.81]) | PREPARED FOOD (0.63 [0.55, 0.71]) | CONDIMENTS/SAUCES (88.6 % / 98.8 %) |

### Complete Journey: fine rankings, eval target baskets (44 of 272 categories eligible)

| # | volume (new events; share) | propensity (rate [CI]; new events) | lift top (lift [CI]) | lift bottom (lift [CI]) | breadth (all users / buyers) |
| --- | --- | --- | --- | --- | --- |
| 1 | BAG SNACKS (492; 3.3 %) | CHRISTMAS  SEASONAL (0.959 [0.919, 0.993]; n 164) | CHRISTMAS  SEASONAL (1.83 [1.74, 1.90]) | FLUID MILK PRODUCTS (0.35 [0.31, 0.40]) | CHEESE (16.2 % / 16.8 %) |
| 2 | CHEESE (471; 3.1 %) | GREETING CARDS/WRAP/PARTY SPLY (0.862 [0.801, 0.915]; n 150) | GREETING CARDS/WRAP/PARTY SPLY (1.64 [1.53, 1.75]) | BAKED BREAD/BUNS/ROLLS (0.67 [0.61, 0.73]) | BAG SNACKS (16.1 % / 16.9 %) |
| 3 | SOFT DRINKS (432; 2.9 %) | SPICES & EXTRACTS (0.805 [0.732, 0.872]; n 128) | SPICES & EXTRACTS (1.53 [1.39, 1.66]) | SOFT DRINKS (0.71 [0.64, 0.77]) | SOFT DRINKS (13.6 % / 13.9 %) |
| 4 | CANDY - PACKAGED (387; 2.6 %) | CANDY - PACKAGED (0.793 [0.750, 0.830]; n 387) | CANDY - PACKAGED (1.51 [1.42, 1.59]) | MILK BY-PRODUCTS (0.72 [0.62, 0.82]) | BAKED BREAD/BUNS/ROLLS (12.2 % / 12.5 %) |
| 5 | FRZN MEAT/MEAT DINNERS (365; 2.4 %) | BAKING NEEDS (0.787 [0.737, 0.841]; n 281) | BAKING NEEDS (1.50 [1.40, 1.60]) | REFRGRATD JUICES/DRNKS (0.72 [0.62, 0.83]) | CANDY - PACKAGED (10.2 % / 11.4 %) |
| 6 | SOUP (350; 2.3 %) | DRY MIX DESSERTS (0.759 [0.592, 0.869]; n 107) | DRY MIX DESSERTS (1.45 [1.13, 1.65]) | YOGURT (0.73 [0.61, 0.86]) | SOUP (9.8 % / 10.7 %) |
| 7 | BAKED BREAD/BUNS/ROLLS (314; 2.1 %) | FD WRAPS/BAGS/TRSH BG (0.696 [0.618, 0.771]; n 110) | FD WRAPS/BAGS/TRSH BG (1.33 [1.18, 1.46]) | WATER - CARBONATED/FLVRD DRINK (0.86 [0.71, 0.99]) | COLD CEREAL (9.8 % / 11.1 %) |
| 8 | BAKING NEEDS (281; 1.9 %) | BAKING MIXES (0.677 [0.605, 0.746]; n 149) | BAKING MIXES (1.29 [1.15, 1.42]) | DELI MEATS (0.86 [0.73, 0.98]) | BEEF (9.1 % / 10.0 %) |
| 9 | FROZEN PIZZA (279; 1.9 %) | COOKIES/CONES (0.660 [0.586, 0.726]; n 138) | COOKIES/CONES (1.26 [1.12, 1.38]) | BEEF (0.88 [0.79, 0.98]) | FRZN MEAT/MEAT DINNERS (8.4 % / 9.8 %) |
| 10 | VEGETABLES - SHELF STABLE (258; 1.7 %) | PNT BTR/JELLY/JAMS (0.653 [0.579, 0.728]; n 113) | PNT BTR/JELLY/JAMS (1.24 [1.11, 1.38]) | CHEESE (0.94 [0.88, 1.00]) | LUNCHMEAT (8.4 % / 9.5 %) |

### Complete Journey: predictability of target new items, coarse level (eligible categories by target volume)

| category | target new items | share of target new items | E2 hit rate [95 % CI] | popularity hit rate | E2 slot share | slot share - target share | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GROCERY | 9,705 | 64.4 % | 0.013 [0.011, 0.016] | 0.014 | 53.3 % | -11.1 pp | 0.83 |
| DRUG GM | 2,310 | 15.3 % | 0.000 [0.000, 0.001] | 0.000 | 1.0 % | -14.3 pp | 0.07 |
| PRODUCE | 887 | 5.9 % | 0.086 [0.066, 0.106] | 0.030 | 36.4 % | +30.5 pp | 6.19 |
| MEAT-PCKGD | 655 | 4.3 % | 0.003 [0.000, 0.008] | 0.000 | 1.5 % | -2.9 pp | 0.34 |
| MEAT | 463 | 3.1 % | 0.028 [0.013, 0.044] | 0.009 | 6.9 % | +3.8 pp | 2.24 |
| DELI | 329 | 2.2 % | 0.000 [0.000, 0.000] | 0.003 | 0.2 % | -2.0 pp | 0.10 |
| PASTRY | 250 | 1.7 % | 0.004 [0.000, 0.014] | 0.000 | 0.3 % | -1.3 pp | 0.20 |
| NUTRITION | 180 | 1.2 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.2 pp | 0.00 |
| SEAFOOD-PCKGD | 91 | 0.6 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.6 pp | 0.02 |
| COSMETICS | 79 | 0.5 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.5 pp | 0.00 |
| FLORAL | 46 | 0.3 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.3 pp | 0.00 |
| SEAFOOD | 35 | 0.2 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.2 pp | 0.00 |
| SALAD BAR | 19 | 0.1 % | 0.000 [0.000, 0.000] | 0.158 | 0.3 % | +0.2 pp | 2.76 |
| SPIRITS | 12 | 0.1 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.1 pp | 0.00 |
| CHEF SHOPPE | 7 | 0.0 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.0 pp | 0.00 |
| FROZEN GROCERY | 3 | 0.0 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.0 pp | 0.00 |
| RESTAURANT | 2 | 0.0 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.0 pp | 0.00 |
| RX | 1 | 0.0 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.0 pp | 0.00 |
| GARDEN CENTER | 1 | 0.0 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.0 pp | 0.00 |
| TRAVEL & LEISUR | 1 | 0.0 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -0.0 pp | 0.00 |

Mismatch (20 categories considered; 3 with an E2 slot share above their target new-item share, 17 below): the largest differences in both directions (rows past the counts are the least negative / least positive):

| # | over-served (slot share vs target share) | under-served (slot share vs target share) |
| --- | --- | --- |
| 1 | PRODUCE (36.4 % vs 5.9 %; +30.5 pp) | DRUG GM (1.0 % vs 15.3 %; -14.3 pp) |
| 2 | MEAT (6.9 % vs 3.1 %; +3.8 pp) | GROCERY (53.3 % vs 64.4 %; -11.1 pp) |
| 3 | SALAD BAR (0.3 % vs 0.1 %; +0.2 pp) | MEAT-PCKGD (1.5 % vs 4.3 %; -2.9 pp) |
| 4 | GARDEN CENTER (0.0 % vs 0.0 %; -0.0 pp) | DELI (0.2 % vs 2.2 %; -2.0 pp) |
| 5 | RX (0.0 % vs 0.0 %; -0.0 pp) | PASTRY (0.3 % vs 1.7 %; -1.3 pp) |
| 6 | TRAVEL & LEISUR (0.0 % vs 0.0 %; -0.0 pp) | NUTRITION (0.0 % vs 1.2 %; -1.2 pp) |
| 7 | RESTAURANT (0.0 % vs 0.0 %; -0.0 pp) | SEAFOOD-PCKGD (0.0 % vs 0.6 %; -0.6 pp) |
| 8 | FROZEN GROCERY (0.0 % vs 0.0 %; -0.0 pp) | COSMETICS (0.0 % vs 0.5 %; -0.5 pp) |
| 9 | CHEF SHOPPE (0.0 % vs 0.0 %; -0.0 pp) | FLORAL (0.0 % vs 0.3 %; -0.3 pp) |
| 10 | SPIRITS (0.0 % vs 0.1 %; -0.1 pp) | SEAFOOD (0.0 % vs 0.2 %; -0.2 pp) |

### Complete Journey: predictability of target new items, fine level (eligible categories by target volume, top 15)

| category | target new items | share of target new items | E2 hit rate [95 % CI] | popularity hit rate | E2 slot share | slot share - target share | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAG SNACKS | 492 | 3.3 % | 0.006 [0.000, 0.013] | 0.000 | 2.2 % | -1.0 pp | 0.69 |
| CHEESE | 471 | 3.1 % | 0.047 [0.030, 0.067] | 0.000 | 6.1 % | +3.0 pp | 1.96 |
| SOFT DRINKS | 432 | 2.9 % | 0.014 [0.003, 0.026] | 0.009 | 4.5 % | +1.6 pp | 1.56 |
| CANDY - PACKAGED | 387 | 2.6 % | 0.000 [0.000, 0.000] | 0.000 | 0.1 % | -2.5 pp | 0.04 |
| FRZN MEAT/MEAT DINNERS | 365 | 2.4 % | 0.003 [0.000, 0.009] | 0.000 | 0.2 % | -2.2 pp | 0.10 |
| SOUP | 350 | 2.3 % | 0.000 [0.000, 0.000] | 0.000 | 0.3 % | -2.1 pp | 0.11 |
| BAKED BREAD/BUNS/ROLLS | 314 | 2.1 % | 0.067 [0.040, 0.097] | 0.115 | 9.7 % | +7.6 pp | 4.65 |
| BAKING NEEDS | 281 | 1.9 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.9 pp | 0.00 |
| FROZEN PIZZA | 279 | 1.9 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.8 pp | 0.01 |
| VEGETABLES - SHELF STABLE | 258 | 1.7 % | 0.012 [0.000, 0.026] | 0.000 | 1.3 % | -0.4 pp | 0.74 |
| COLD CEREAL | 256 | 1.7 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.7 pp | 0.03 |
| LUNCHMEAT | 231 | 1.5 % | 0.000 [0.000, 0.000] | 0.000 | 0.0 % | -1.5 pp | 0.00 |
| BEEF | 219 | 1.5 % | 0.041 [0.016, 0.070] | 0.014 | 5.0 % | +3.5 pp | 3.42 |
| BABY FOODS | 214 | 1.4 % | 0.000 [0.000, 0.000] | 0.000 | 0.1 % | -1.4 pp | 0.04 |
| CRACKERS/MISC BKD FD | 206 | 1.4 % | 0.000 [0.000, 0.000] | 0.000 | 0.9 % | -0.5 pp | 0.65 |

Mismatch (61 categories considered; 26 with an E2 slot share above their target new-item share, 35 below): the largest differences in both directions (rows past the counts are the least negative / least positive):

| # | over-served (slot share vs target share) | under-served (slot share vs target share) |
| --- | --- | --- |
| 1 | EGGS (9.4 % vs 0.5 %; +8.8 pp) | CANDY - PACKAGED (0.1 % vs 2.6 %; -2.5 pp) |
| 2 | BAKED BREAD/BUNS/ROLLS (9.7 % vs 2.1 %; +7.6 pp) | FRZN MEAT/MEAT DINNERS (0.2 % vs 2.4 %; -2.2 pp) |
| 3 | FLUID MILK PRODUCTS (6.1 % vs 1.3 %; +4.8 pp) | SOUP (0.3 % vs 2.3 %; -2.1 pp) |
| 4 | POTATOES (4.3 % vs 0.4 %; +3.9 pp) | BAKING NEEDS (0.0 % vs 1.9 %; -1.9 pp) |
| 5 | ONIONS (4.0 % vs 0.4 %; +3.6 pp) | FROZEN PIZZA (0.0 % vs 1.9 %; -1.8 pp) |
| 6 | BEEF (5.0 % vs 1.5 %; +3.5 pp) | COLD CEREAL (0.0 % vs 1.7 %; -1.7 pp) |
| 7 | VEGETABLES - ALL OTHERS (3.7 % vs 0.5 %; +3.3 pp) | LUNCHMEAT (0.0 % vs 1.5 %; -1.5 pp) |
| 8 | TOMATOES (3.5 % vs 0.3 %; +3.3 pp) | BABY FOODS (0.1 % vs 1.4 %; -1.4 pp) |
| 9 | CHEESE (6.1 % vs 3.1 %; +3.0 pp) | CANNED JUICES (0.0 % vs 1.3 %; -1.3 pp) |
| 10 | VEGETABLES SALAD (3.2 % vs 0.2 %; +3.0 pp) | REFRGRATD DOUGH PRODUCTS (0.0 % vs 1.2 %; -1.2 pp) |

### Complete Journey: 2 x 2 placement of fine categories (44 categories; median rate 0.590, median E2 hit rate 0.000)

- explored but hard to predict (rate above median, E2 hit rate below median): 21 categories: CANDY - PACKAGED (0.759, 0.000, n 387); BAKING NEEDS (0.787, 0.000, n 281); LUNCHMEAT (0.594, 0.000, n 231); REFRGRATD DOUGH PRODUCTS (0.634, 0.000, n 179); CHRISTMAS  SEASONAL (0.959, 0.000, n 164); FRZN VEGETABLE/VEG DSH (0.592, 0.000, n 154); DRY BN/VEG/POTATO/RICE (0.661, 0.000, n 152); CONDIMENTS/SAUCES (0.699, 0.000, n 150); GREETING CARDS/WRAP/PARTY SPLY (0.861, 0.000, n 150); FRUIT - SHELF STABLE (0.621, 0.000, n 144); COOKIES/CONES (0.671, 0.000, n 138); HISPANIC (0.676, 0.000, n 137); CONVENIENT BRKFST/WHLSM SNACKS (0.604, 0.000, n 133); DRY NOODLES/PASTA (0.660, 0.000, n 133); ICE CREAM/MILK/SHERBTS (0.634, 0.000, n 132); DINNER MXS:DRY (0.591, 0.000, n 128); SPICES & EXTRACTS (0.829, 0.000, n 128); SALD DRSNG/SNDWCH SPRD (0.667, 0.000, n 127); PNT BTR/JELLY/JAMS (0.641, 0.000, n 113); FD WRAPS/BAGS/TRSH BG (0.727, 0.000, n 110); DRY MIX DESSERTS (0.766, 0.000, n 107)

- explored and easier to predict (both above median): 1 categories: BAKING MIXES (0.734, 0.007, n 149)

- rarely explored and hard to predict (both below median): 11 categories: SOUP (0.578, 0.000, n 350); FROZEN PIZZA (0.574, 0.000, n 279); COLD CEREAL (0.578, 0.000, n 256); BABY FOODS (0.555, 0.000, n 214); CRACKERS/MISC BKD FD (0.588, 0.000, n 206); CANNED JUICES (0.548, 0.000, n 203); YOGURT (0.433, 0.000, n 185); BAKED SWEET GOODS (0.548, 0.000, n 165); DELI MEATS (0.452, 0.000, n 127); MEAT - SHELF STABLE (0.554, 0.000, n 124); WATER - CARBONATED/FLVRD DRINK (0.437, 0.000, n 103)

- rarely explored but easier to predict (rate below, E2 above): 11 categories: BAG SNACKS (0.555, 0.006, n 492); CHEESE (0.524, 0.047, n 471); SOFT DRINKS (0.384, 0.014, n 432); FRZN MEAT/MEAT DINNERS (0.587, 0.003, n 365); BAKED BREAD/BUNS/ROLLS (0.394, 0.067, n 314); VEGETABLES - SHELF STABLE (0.577, 0.012, n 258); BEEF (0.538, 0.041, n 219); FLUID MILK PRODUCTS (0.178, 0.113, n 195); CANDY - CHECKLANE (0.563, 0.007, n 134); REFRGRATD JUICES/DRNKS (0.390, 0.008, n 127); MILK BY-PRODUCTS (0.462, 0.008, n 124)

### Complete Journey: robustness

Spearman rank correlation of the fit-split and eval-split category values (history + target events; eligible eval categories present in both). The fit split shares most of its events with the eval split (Instacart: every basket but the last; Complete Journey: an earlier cutoff), so these correlations check the stability of the category values rather than provide independent evidence:

| level | categories | volume (new-item events) | propensity (rate) |
| --- | --- | --- | --- |
| coarse | 30 | 1.000 | 0.997 |
| fine | 266 | 0.999 | 0.999 |

Coarse new-item rate by T5 history-length bin (eval, history + target events; categories sorted by volume; '-' = no events in the bin):

| category | (0,20] (138 users) | (20,50] (547 users) | (50,100] (653 users) | (100,inf) (731 users) |
| --- | --- | --- | --- | --- |
| overall | 0.826 | 0.702 | 0.587 | 0.493 |
| GROCERY | 0.827 | 0.693 | 0.575 | 0.479 |
| DRUG GM | 0.887 | 0.841 | 0.764 | 0.659 |
| PRODUCE | 0.720 | 0.580 | 0.445 | 0.350 |
| MEAT-PCKGD | 0.860 | 0.731 | 0.626 | 0.525 |
| MEAT | 0.860 | 0.729 | 0.611 | 0.524 |
| DELI | 0.800 | 0.667 | 0.553 | 0.439 |
| NUTRITION | 0.801 | 0.726 | 0.531 | 0.521 |
| PASTRY | 0.866 | 0.773 | 0.704 | 0.596 |
| SEAFOOD-PCKGD | 0.860 | 0.750 | 0.743 | 0.648 |
| COSMETICS | 0.947 | 0.935 | 0.904 | 0.887 |
| FLORAL | 1.000 | 0.945 | 0.905 | 0.827 |
| SALAD BAR | 0.703 | 0.525 | 0.360 | 0.259 |
| SEAFOOD | 0.923 | 0.842 | 0.713 | 0.561 |
| SPIRITS | 0.591 | 0.947 | 0.787 | 0.486 |
| GARDEN CENTER | 1.000 | 1.000 | 0.934 | 0.861 |
| DAIRY DELI | - | 1.000 | 1.000 | 0.926 |
| TRAVEL & LEISUR | 1.000 | 0.802 | 0.674 | 0.662 |
| CHEF SHOPPE | 1.000 | 0.790 | 0.736 | 0.640 |
| FROZEN GROCERY | 0.833 | 0.857 | 0.855 | 0.819 |
| PHOTO | - | 1.000 | 1.000 | 1.000 |
| GRO BAKERY | - | - | - | 1.000 |
| POSTAL CENTER | - | 1.000 | 1.000 | 1.000 |
| RX | - | 1.000 | 1.000 | 1.000 |
| PORK | - | - | 1.000 | - |
| PROD-WHS SALES | - | - | 0.333 | 1.000 |
| TOYS | - | - | 1.000 | 0.667 |
| VIDEO | - | - | - | 1.000 |
| HOUSEWARES | - | - | 1.000 | - |
| DELI/SNACK BAR | - | 1.000 | 1.000 | 0.800 |
| HBC | - | 1.000 | 1.000 | 1.000 |
| RESTAURANT | 1.000 | 0.500 | 0.647 | 0.538 |

## Figures

- F-1 `fig_category_rate_coarse.png` (`fig_category_rate_coarse.csv`): coarse categories, new-item rate with 95 % CI, labelled with the share of all new-item events, one panel per dataset.
- F-2 `fig_category_lift_fine.png` (`fig_category_lift_fine.csv`): fine categories above the support filter, lift on a log scale, top 15 and bottom 10, one panel per dataset.
- F-3 `fig_category_rate_vs_e2_fine.png` (`fig_category_rate_vs_e2_fine.csv`): fine categories of the 2 x 2 placement, new-item rate vs E2 hit rate, point size = target new items, median lines, top 8 by volume labelled.

## Timing and memory

| dataset | eval load | events | category tables + CIs | E2 list | popularity list | predictability | robustness | peak working set (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Instacart | 6.9 s | 4.9 s | 13.3 s | 81.5 s | 10.6 s | 2.0 s | 15.6 s | 3469 |
| Complete Journey | 5.1 s | 2.9 s | 8.3 s | 27.7 s | 6.1 s | 0.5 s | 9.3 s | 3469 |
