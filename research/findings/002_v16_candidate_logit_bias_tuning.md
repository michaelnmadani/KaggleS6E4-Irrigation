# V16 research — logit-bias tuning for balanced accuracy

## Hypothesis
V11's named best model is `logit_bias_tuned` — a lightweight post-processing step applied to the blended probabilities that tunes per-class logit offsets so balanced-accuracy on OOF is maximized. On V11's data, this took the raw blend from ~0.9691 to **0.97256** (+0.005+).

## Why it works on this data
Class prior is very skewed (Low 58.7%, Medium 38.0%, **High 3.3%**). Soft-label argmax on unbalanced data systematically under-predicts the minority class even when training loss is reasonable. Shifting logits per class trades a few majority-class errors for many more minority-class wins — exactly what balanced accuracy rewards.

## Expected gain
- **+0.003 to +0.006** balanced-accuracy on top of whatever the blended base model scores.
- Gain is biggest on top of already-balanced models (V14's class-weighted blend). On unweighted models, gain can be as high as +0.01.

## Algorithm (nested CV, no additional training)
```python
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

def tune_logit_bias(oof_probs, y_true, n_classes=3):
    # logit = log(probs), add bias vector b, softmax, argmax, score balanced_acc.
    def neg_ba(b):
        logits = np.log(np.clip(oof_probs, 1e-9, 1)) + b
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        preds = probs.argmax(1)
        return -balanced_accuracy_score(y_true, preds)
    res = minimize(neg_ba, x0=np.zeros(n_classes), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-5})
    return res.x  # bias vector to add to test logits
```

Fit on OOF probabilities (no leakage since OOF already respects fold split). Apply bias to test probabilities before argmax.

## Risks
- **OOF overfitting**: V10 explicitly removed probability calibration because it widened the CV-LB gap (V9 had CV 0.96999 vs LB 0.96715, -0.00284). Logit-bias is simpler (3 parameters) but the same risk applies.
- **Mitigation**: V11 used **nested CV** for bias tuning — hold out one fold at a time, compute bias on the other 9, apply to the held-out fold. This gives an unbiased CV estimate of the post-bias score.
- **Hyperparameter sensitivity**: optimizer tolerance matters. Too tight = overfits OOF noise; too loose = leaves gains on the table. Nelder-Mead with xatol=1e-4, fatol=1e-5 is a known-good setting from V11.

## Implementation sketch (delta from V15)
- `pipeline/src/postprocess.py` (new) — `tune_logit_bias(oof, y)` → bias vector; `apply_logit_bias(probs, bias)` → new probs.
- `pipeline/src/train.py` — after CV loop, if `cfg.postprocess == "logit_bias"`, run nested-CV bias fit on blended OOF, apply bias to `test_preds` before the argmax in the submission writer, and report both the raw and post-bias balanced-accuracy in `metrics.json`.
- `iterations/005_logit_bias/config.yaml` — add `postprocess: logit_bias` knob on top of V15's config.

## Decision gates
- Run V16 only after V14 and V15 have landed and we know the raw blend's balanced-acc. If V15 already hits ≥0.972, V16 may over-tune; skip.
- If V16's OOF balanced-acc is >+0.003 above raw blend, submit. If OOF gain is <0.001, don't submit (overfitting signal).

## References
- V11 results.json: "Logit-space bias tuning with nested CV: per-class logit offsets for balanced accuracy"
- V10 results.json: "Remove probability calibration entirely (source of CV-LB gap: 0.96999 CV vs 0.96715 LB)" — cautionary tale about naive calibration
