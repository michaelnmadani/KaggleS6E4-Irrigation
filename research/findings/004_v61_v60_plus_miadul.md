# V61 research — V60 (LB champion) + miadul external augmentation

## Hypothesis
V60 broke the V34 LB plateau by under-fitting more (real ES at 80 rounds, depth=5)
and giving Optuna a larger postprocess delta (+0.00382 vs V34's +0.00125). The
under-fit raw model has more noise the external miadul rows could *replace* with
clean labels — cdeotte hits CV bal-acc 1.0 on miadul with 4 threshold rules, so
miadul's labels are well-separated.

V35 tried miadul on V34's recipe and got LB 0.97517 (-0.001 vs V34 LB 0.97617). But
V35 also flipped stratified=true, conflating two changes. V61 isolates the miadul
effect by keeping every other V60 hyperparameter identical (stratified=false, fold
seed=42, 1000 rounds, ES=80, depth=5, lr=0.07).

## Expected outcome
- miadul rows align with synthetic train distribution (comp data was synthesized FROM miadul) → adding 10k clean rows could improve recall on Medium class (V60's weakest at 0.962)
- Risk: miadul rows that resemble test rows leak. data.py already dedupes by 6-decimal numeric hash.
- Expected CV: 0.976 ± 0.002. LB: ±0.001 vs V60.

## Decision gate
- If V61 LB > V60 LB (0.97646): scale up — V62-V64 sweep weight, V65 bag with 3 seeds
- If V61 LB < V60 LB by >0.0005: stop the miadul thread, V62 pivots to a different angle
