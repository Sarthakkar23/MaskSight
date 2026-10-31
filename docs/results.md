## Main results

| Method | Image AUROC | Pixel AUROC | AUPRO | Dice: quantile | Dice: synth-abs | **Dice: SynthCal** | Dice: oracle | Image F1 (label-free) | Latency (T4) |
|---|---|---|---|---|---|---|---|---|---|
| padim | 0.986 | 0.975 | 0.895 | 0.366 | 0.321 | **0.480** | 0.432 | 0.893 | 16 ms |
| patchcore | 0.990 | 0.978 | 0.907 | 0.397 | 0.369 | **0.517** | 0.466 | 0.882 | 29 ms |

- padim: SynthCal beats the quantile baseline on 15/15 categories and matches or beats the oracle on 11/15.
- patchcore: SynthCal beats the quantile baseline on 15/15 categories and matches or beats the oracle on 13/15.

<details><summary>Per-category results</summary>

| Category | padim: img AUROC | patchcore: img AUROC | padim: AUPRO | patchcore: AUPRO | padim: Dice q | patchcore: Dice q | padim: **Dice SC** | patchcore: **Dice SC** | padim: Dice oracle | patchcore: Dice oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| bottle | 1.000 | 1.000 | 0.905 | 0.926 | 0.471 | 0.513 | 0.696 | 0.729 | 0.673 | 0.707 |
| cable | 0.993 | 0.996 | 0.913 | 0.918 | 0.562 | 0.584 | 0.609 | 0.624 | 0.555 | 0.577 |
| capsule | 0.939 | 0.972 | 0.902 | 0.909 | 0.333 | 0.326 | 0.369 | 0.377 | 0.230 | 0.283 |
| carpet | 0.995 | 0.988 | 0.908 | 0.922 | 0.220 | 0.286 | 0.509 | 0.529 | 0.411 | 0.457 |
| grid | 0.981 | 0.987 | 0.900 | 0.902 | 0.232 | 0.223 | 0.352 | 0.349 | 0.280 | 0.352 |
| hazelnut | 1.000 | 1.000 | 0.869 | 0.859 | 0.406 | 0.444 | 0.564 | 0.577 | 0.497 | 0.523 |
| leather | 1.000 | 1.000 | 0.955 | 0.967 | 0.088 | 0.122 | 0.382 | 0.425 | 0.289 | 0.347 |
| metal_nut | 1.000 | 0.998 | 0.882 | 0.920 | 0.574 | 0.657 | 0.613 | 0.685 | 0.662 | 0.684 |
| pill | 0.965 | 0.960 | 0.933 | 0.919 | 0.387 | 0.444 | 0.400 | 0.450 | 0.417 | 0.443 |
| screw | 0.954 | 0.985 | 0.934 | 0.950 | 0.190 | 0.239 | 0.268 | 0.326 | 0.239 | 0.247 |
| tile | 0.992 | 0.990 | 0.771 | 0.812 | 0.473 | 0.564 | 0.502 | 0.647 | 0.507 | 0.573 |
| toothbrush | 1.000 | 0.997 | 0.853 | 0.863 | 0.242 | 0.308 | 0.389 | 0.426 | 0.276 | 0.295 |
| transistor | 1.000 | 1.000 | 0.949 | 0.947 | 0.511 | 0.500 | 0.514 | 0.511 | 0.584 | 0.528 |
| wood | 0.995 | 0.990 | 0.836 | 0.857 | 0.296 | 0.319 | 0.458 | 0.491 | 0.365 | 0.424 |
| zipper | 0.976 | 0.992 | 0.915 | 0.925 | 0.511 | 0.425 | 0.574 | 0.604 | 0.497 | 0.551 |

</details>

## Ablations

Averaged over bottle, carpet, hazelnut, leather, screw (PatchCore).

| Variant | Image AUROC | AUPRO | Dice: quantile | **Dice: SynthCal** | Dice: oracle | Fit time | Latency |
|---|---|---|---|---|---|---|---|
| baseline (σ=4, 10%, layer2+3) | 0.995 | 0.925 | 0.321 | **0.517** | 0.456 | 58 s | 32 ms |
| sigma0 | 0.993 | 0.926 | 0.334 | **0.518** | 0.465 | 58 s | 31 ms |
| sigma2 | 0.994 | 0.926 | 0.330 | **0.520** | 0.464 | 57 s | 32 ms |
| sigma8 | 0.995 | 0.919 | 0.294 | **0.499** | 0.436 | 57 s | 36 ms |
| coreset1pct | 0.992 | 0.921 | 0.311 | **0.514** | 0.456 | 15 s | 15 ms |
| coreset25pct | 0.995 | 0.924 | 0.321 | **0.514** | 0.452 | 128 s | 63 ms |
| layer2only | 0.988 | 0.942 | 0.427 | **0.554** | 0.530 | 57 s | 14 ms |
| resnet18 | 0.990 | 0.921 | 0.346 | **0.523** | 0.461 | 56 s | 10 ms |
