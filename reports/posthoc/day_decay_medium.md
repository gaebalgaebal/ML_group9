# Day-based recency decay for the repeat scorer (medium)

**post-hoc, development data, not confirmatory.** Repeat Recall@10 on users with >= 1 repeat target item; day decay score(u, i) = sum over history baskets containing i of lam ** (d_last(u) - d_b), lam = 2 ** (-1/h), h = half-life in days (h = inf is personal frequency); basket decay = nbr.repeat with gamma per basket age. Selections are made on the fit split; the eval split (already used by earlier development runs) carries paired-bootstrap CIs only. No confirmatory claim is made.

- Code commit `8706f7920a1822317280a83646b90825831e874c`, 2026-11-06T10:01:51.572139+00:00; profile `medium`; bootstrap n = 1000, seed 42, 95% CI.
- Grids: h in days [7, 14, 30, 60, 90, 180, 365, inf]; gamma [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0].
- Inputs (sha256 in the JSON): instacart: 5 files, complete_journey: 5 files.
- Runtime 302.8 s; peak working set 2438 MiB (in-process).

## Data and metric

| dataset | fit users (repeat target / all) | eval users (repeat target / all) | fit baskets | eval baskets | inter-basket gap mean / median (days, fit) | share of zero gaps |
| --- | --- | --- | --- | --- | --- | --- |
| Instacart | 18,510 / 20,000 | 18,643 / 20,000 | 293,281 | 313,281 | 10.24 / 7 | 0.022 |
| Complete Journey | 1,828 / 2,069 | 1,867 / 2,069 | 180,368 | 206,682 | 5.55 / 3 | 0.160 |

## (a) Fit-split grids and selections

Day decay (h in days; * = h*):

| h (days) | Instacart fit recall | Complete Journey fit recall |
| --- | --- | --- |
| 7 | 0.5594 | 0.1996 |
| 14 | 0.5705 | 0.2230 |
| 30 | 0.5820 | 0.2526 |
| 60 | 0.5844 * | 0.2722 |
| 90 | 0.5811 | 0.2797 |
| 180 | 0.5729 | 0.2810 * |
| 365 | 0.5680 | 0.2795 |
| inf | 0.5594 | 0.2766 |

Basket decay (gamma; half-life in baskets = ln 0.5 / ln gamma; * = gamma*):

| gamma | half-life (baskets) | Instacart fit recall | Complete Journey fit recall |
| --- | --- | --- | --- |
| 0.5 | 1.0 | 0.5535 | 0.1853 |
| 0.6 | 1.4 | 0.5639 | 0.1923 |
| 0.7 | 1.9 | 0.5769 | 0.2055 |
| 0.8 | 3.1 | 0.5834 * | 0.2321 |
| 0.9 | 6.6 | 0.5823 | 0.2740 |
| 0.95 | 13.5 | 0.5756 | 0.2862 |
| 0.97 | 22.8 | 0.5714 | 0.2890 * |
| 0.98 | 34.3 | 0.5690 | 0.2848 |
| 0.99 | 69.0 | 0.5668 | 0.2806 |
| 0.995 | 138.3 | 0.5659 | 0.2772 |
| 1 | inf | 0.5594 | 0.2766 |

| dataset | h* (days) | h* at grid edge | gamma* | gamma* at grid edge |
| --- | --- | --- | --- | --- |
| Instacart | 60 | False | 0.8 | False |
| Complete Journey | 180 | False | 0.97 | False |

## (b) Eval-split comparisons (paired bootstrap over users; diff = model - comparator)

| dataset | model | comparator | comparator mean | model mean | diff | 95 % CI | users |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Instacart | day decay h* = 60 days | basket decay gamma = 1 (personal frequency) | 0.5345 | 0.5647 | +0.0302 | [+0.0276, +0.0333] | 18,643 |
| Instacart | day decay h* = 60 days | basket decay gamma* = 0.8 | 0.5630 | 0.5647 | +0.0017 | [+0.0002, +0.0032] | 18,643 |
| Instacart | basket decay gamma* = 0.8 | basket decay gamma = 1 (personal frequency) | 0.5345 | 0.5630 | +0.0285 | [+0.0255, +0.0318] | 18,643 |
| Complete Journey | day decay h* = 180 days | basket decay gamma = 1 (personal frequency) | 0.2497 | 0.2645 | +0.0148 | [+0.0068, +0.0226] | 1,867 |
| Complete Journey | day decay h* = 180 days | basket decay gamma* = 0.97 | 0.2717 | 0.2645 | -0.0072 | [-0.0137, -0.0011] | 1,867 |
| Complete Journey | basket decay gamma* = 0.97 | basket decay gamma = 1 (personal frequency) | 0.2497 | 0.2717 | +0.0220 | [+0.0142, +0.0299] | 1,867 |

Eval-split recall of every grid point (reference only):

| h (days) | Instacart eval recall | Complete Journey eval recall |
| --- | --- | --- |
| 7 | 0.5356 | 0.1942 |
| 14 | 0.5463 | 0.2131 |
| 30 | 0.5621 | 0.2419 |
| 60 | 0.5647 | 0.2624 |
| 90 | 0.5599 | 0.2675 |
| 180 | 0.5513 | 0.2645 |
| 365 | 0.5458 | 0.2577 |
| inf | 0.5345 | 0.2497 |

| gamma | Instacart eval recall | Complete Journey eval recall |
| --- | --- | --- |
| 0.5 | 0.5310 | 0.1785 |
| 0.6 | 0.5426 | 0.1845 |
| 0.7 | 0.5555 | 0.1993 |
| 0.8 | 0.5630 | 0.2193 |
| 0.9 | 0.5611 | 0.2579 |
| 0.95 | 0.5533 | 0.2748 |
| 0.97 | 0.5494 | 0.2717 |
| 0.98 | 0.5469 | 0.2634 |
| 0.99 | 0.5446 | 0.2567 |
| 0.995 | 0.5434 | 0.2523 |
| 1 | 0.5345 | 0.2497 |

## (c) Transfer of the selection between datasets (eval split; loss = own-selection recall - transferred recall)

| dataset | scorer | own selection | transferred selection | own recall | transferred recall | transfer loss | 95 % CI | users |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Instacart | day | h = 60 | h = 180 (from Complete Journey) | 0.5647 | 0.5513 | +0.0134 | [+0.0118, +0.0151] | 18,643 |
| Instacart | basket | gamma = 0.8 | gamma = 0.97 (from Complete Journey) | 0.5630 | 0.5494 | +0.0137 | [+0.0115, +0.0159] | 18,643 |
| Complete Journey | day | h = 180 | h = 60 (from Instacart) | 0.2645 | 0.2624 | +0.0020 | [-0.0059, +0.0095] | 1,867 |
| Complete Journey | basket | gamma = 0.97 | gamma = 0.8 (from Instacart) | 0.2717 | 0.2193 | +0.0523 | [+0.0417, +0.0635] | 1,867 |

## (d) Unified scale: basket half-life and its day equivalent

Instacart: mean gap 10.24 d, median gap 7 d; h* = 60 d = 5.9 baskets (mean gap) = 8.6 baskets (median gap); gamma* = 0.8.

| gamma (* = gamma*) | half-life (baskets) | day equivalent, mean gap | day equivalent, median gap |
| --- | --- | --- | --- |
| 0.5 | 1.0 | 10.2 | 7.0 |
| 0.6 | 1.4 | 13.9 | 9.5 |
| 0.7 | 1.9 | 19.9 | 13.6 |
| 0.8 * | 3.1 | 31.8 | 21.7 |
| 0.9 | 6.6 | 67.4 | 46.1 |
| 0.95 | 13.5 | 138.4 | 94.6 |
| 0.97 | 22.8 | 233.1 | 159.3 |
| 0.98 | 34.3 | 351.5 | 240.2 |
| 0.99 | 69.0 | 706.5 | 482.8 |
| 0.995 | 138.3 | 1416.5 | 968.0 |
| 1 | inf | inf | inf |
| h* = 60 d | 8.6 (median gap) | 60 | 60 |

Complete Journey: mean gap 5.55 d, median gap 3 d; h* = 180 d = 32.4 baskets (mean gap) = 60.0 baskets (median gap); gamma* = 0.97.

| gamma (* = gamma*) | half-life (baskets) | day equivalent, mean gap | day equivalent, median gap |
| --- | --- | --- | --- |
| 0.5 | 1.0 | 5.5 | 3.0 |
| 0.6 | 1.4 | 7.5 | 4.1 |
| 0.7 | 1.9 | 10.8 | 5.8 |
| 0.8 | 3.1 | 17.2 | 9.3 |
| 0.9 | 6.6 | 36.5 | 19.7 |
| 0.95 | 13.5 | 75.0 | 40.5 |
| 0.97 * | 22.8 | 126.3 | 68.3 |
| 0.98 | 34.3 | 190.3 | 102.9 |
| 0.99 | 69.0 | 382.6 | 206.9 |
| 0.995 | 138.3 | 767.2 | 414.8 |
| 1 | inf | inf | inf |
| h* = 180 d | 60.0 (median gap) | 180 | 180 |

## (e) History-length bins (distinct fit-history baskets; fit recall)

Instacart, day decay by h (days):

| bin | users | 7 | 14 | 30 | 60 | 90 | 180 | 365 | inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,920 | 0.7125 | 0.7142 | 0.7172 | 0.7213 | 0.7213 | 0.7213 | 0.7213 | 0.7215 |
| (4,10] | 5,435 | 0.5718 | 0.5805 | 0.5899 | 0.5947 | 0.5922 | 0.5882 | 0.5872 | 0.5779 |
| (10,20] | 3,905 | 0.4832 | 0.4960 | 0.5150 | 0.5201 | 0.5159 | 0.5044 | 0.4976 | 0.4852 |
| (20,50] | 3,276 | 0.4410 | 0.4619 | 0.4777 | 0.4743 | 0.4669 | 0.4474 | 0.4346 | 0.4205 |
| (50,100] | 974 | 0.4195 | 0.4524 | 0.4748 | 0.4644 | 0.4554 | 0.4341 | 0.4177 | 0.4027 |
| (100,inf) | 0 | - | - | - | - | - | - | - | - |

Instacart, basket decay by gamma:

| bin | users | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 0.97 | 0.98 | 0.99 | 0.995 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 4,920 | 0.7117 | 0.7133 | 0.7209 | 0.7213 | 0.7213 | 0.7213 | 0.7213 | 0.7213 | 0.7213 | 0.7213 | 0.7215 |
| (4,10] | 5,435 | 0.5690 | 0.5789 | 0.5891 | 0.5938 | 0.5889 | 0.5871 | 0.5871 | 0.5871 | 0.5871 | 0.5871 | 0.5779 |
| (10,20] | 3,905 | 0.4780 | 0.4921 | 0.5101 | 0.5189 | 0.5148 | 0.5033 | 0.4991 | 0.4973 | 0.4971 | 0.4971 | 0.4852 |
| (20,50] | 3,276 | 0.4315 | 0.4492 | 0.4654 | 0.4761 | 0.4750 | 0.4572 | 0.4432 | 0.4358 | 0.4285 | 0.4261 | 0.4205 |
| (50,100] | 974 | 0.3817 | 0.3987 | 0.4248 | 0.4492 | 0.4753 | 0.4632 | 0.4477 | 0.4352 | 0.4175 | 0.4087 | 0.4027 |
| (100,inf) | 0 | - | - | - | - | - | - | - | - | - | - | - |

Complete Journey, day decay by h (days):

| bin | users | 7 | 14 | 30 | 60 | 90 | 180 | 365 | inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 3 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| (4,10] | 35 | 0.3236 | 0.3522 | 0.3665 | 0.3605 | 0.3891 | 0.4181 | 0.4228 | 0.5801 |
| (10,20] | 104 | 0.2379 | 0.2432 | 0.2610 | 0.2811 | 0.3144 | 0.3382 | 0.3437 | 0.3709 |
| (20,50] | 490 | 0.2171 | 0.2273 | 0.2494 | 0.2791 | 0.2995 | 0.3098 | 0.3201 | 0.3101 |
| (50,100] | 658 | 0.2018 | 0.2239 | 0.2634 | 0.2837 | 0.2817 | 0.2773 | 0.2728 | 0.2661 |
| (100,inf) | 538 | 0.1620 | 0.2024 | 0.2301 | 0.2412 | 0.2423 | 0.2364 | 0.2260 | 0.2177 |

Complete Journey, basket decay by gamma:

| bin | users | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 0.97 | 0.98 | 0.99 | 0.995 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (0,4] | 3 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| (4,10] | 35 | 0.3236 | 0.3522 | 0.3510 | 0.4244 | 0.4514 | 0.4371 | 0.4371 | 0.4371 | 0.4371 | 0.4371 | 0.5801 |
| (10,20] | 104 | 0.2418 | 0.2504 | 0.2517 | 0.2992 | 0.3495 | 0.3323 | 0.3407 | 0.3417 | 0.3417 | 0.3417 | 0.3709 |
| (20,50] | 490 | 0.2135 | 0.2186 | 0.2348 | 0.2554 | 0.2987 | 0.3184 | 0.3242 | 0.3159 | 0.3146 | 0.3136 | 0.3101 |
| (50,100] | 658 | 0.1932 | 0.2008 | 0.2140 | 0.2407 | 0.2765 | 0.2833 | 0.2815 | 0.2754 | 0.2726 | 0.2680 | 0.2661 |
| (100,inf) | 538 | 0.1263 | 0.1329 | 0.1463 | 0.1716 | 0.2193 | 0.2386 | 0.2433 | 0.2440 | 0.2342 | 0.2293 | 0.2177 |

## Reproduction check

| dataset | quantity | expected | observed | matched at 4 decimals |
| --- | --- | --- | --- | --- |
| Complete Journey | fit_basket_0.97 | 0.2890 | 0.2890 | yes |
| Complete Journey | fit_basket_1.0 | 0.2766 | 0.2766 | yes |
| Complete Journey | eval_basket_0.97_minus_1.0 | 0.0220 | 0.0220 | yes |

## Figure

`fig_recall_vs_halflife.png` (numbers in `fig_recall_vs_halflife.csv`): fit repeat Recall@10 vs half-life in days (log x); day decay solid, basket decay converted with the median inter-basket gap dashed; one panel per dataset; dotted line = personal frequency (h = inf, gamma = 1).

## Timing and memory

| dataset | load + check | fit grids | eval grids | bootstrap | gaps + bins | peak working set (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| Instacart | 8.6 s | 87.4 s | 91.9 s | 0.4 s | 2.3 s | 2438 |
| Complete Journey | 5.5 s | 49.4 s | 55.0 s | 0.0 s | 1.3 s | 2438 |
