"""RealMLP-TD classifier — parallel-ensemble MLP with periodic numeric
embeddings and heavy regularization. Architecture mirrors pytabkit's RealMLP
and matches the PS-S6E4 reference notebook (CV 0.978, LB 0.977).

Ported verbatim from mahoganybuttstrings/pg-s6e4-realmlp-cv-0-97802-lb-0-97685.
Runs on a Kaggle GPU kernel; fails gracefully to CPU if cuda is unavailable
(much slower — avoid).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight


REALMLP_DEFAULT_CONFIG = {
    "n_ens": 8,
    "embed_dim": 6,
    "onehot_thresh": 8,
    "hidden_dims": [512, 64, 128],
    "dropout": 0.05,
    "activation": "silu",
    "add_front_scale": True,
    "pbld_hidden_dim": 20,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 1.0,
    "pbld_activation": "gelu",
    "pbld_lr_factor": 0.1,
    "lr": 1e-2,
    "mom": 0.9,
    "sq_mom": 0.98,
    "lr_sched": "flat_cos",
    "flat_ratio": 0.3,
    "first_layer_lr_factor": 1.0,
    "lr_scale_mult": 10.0,
    "lr_bias_mult": 0.1,
    "weight_decay": 1e-2,
    "wd_scale_mult": 0.1,
    "wd_bias_mult": 0.5,
    "grad_clip": 1.0,
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    "p_drop_sched": "expm4t",
    "use_early_stopping": False,
    "early_stopping_additive_patience": 20,
    "early_stopping_multiplicative_patience": 2.0,
    "epochs": 3,
    "train_bs": 256,
    "eval_bs": 10240,
    "verbosity": 1,
    "tfms": ["median_center", "robust_scale", "smooth_clip"],
    "device": "cuda",
    "random_state": 42,
}


_ACT_MAP = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU, "elu": nn.ELU}


def _resolve_activation(name):
    if isinstance(name, str):
        return _ACT_MAP.get(name.lower(), nn.SiLU)
    return name


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [t for t in tfms if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)])
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx:feat_idx + 1].long()
                feat_embs.append(emb_list[model_idx](indices))
            feat_combined = torch.cat(feat_embs, dim=1)
            features.append(feat_combined)
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLPNet(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim, onehot_thresh=cfg["onehot_thresh"])
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens, n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"], activation=_resolve_activation(cfg["pbld_activation"]),
        )
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]
        act = _resolve_activation(cfg["activation"])
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(hidden_dims):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return F.softmax(x, dim=2)


def _apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    if sched == "constant":
        return init_value
    if sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    if sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    if sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    if sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    if sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    raise ValueError(f"Unknown schedule: {sched!r}")


def _parameter_groups(model, p):
    first_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    LR, WD = p["lr"], p["weight_decay"]
    return [
        {"params": scale_p,   "lr": LR * p["lr_scale_mult"],         "weight_decay": WD * p["wd_scale_mult"]},
        {"params": pbld_p,    "lr": LR * p["pbld_lr_factor"],        "weight_decay": WD},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD},
        {"params": other_w_p, "lr": LR,                               "weight_decay": WD},
        {"params": bias_p,    "lr": LR * p["lr_bias_mult"],          "weight_decay": WD * p["wd_bias_mult"]},
    ]


def _smooth_ce(y_true, y_pred, ls=0.0, class_weights=None):
    n_classes = y_pred.size(1)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if class_weights is not None:
        sw = class_weights[y_true]
        return (per_sample * sw).sum() / sw.sum()
    return per_sample.mean()


class RealMLPClassifier:
    """Thin wrapper exposing .fit + .predict_proba compatible with our FitResult."""

    def __init__(self, **overrides):
        self.p = {**REALMLP_DEFAULT_CONFIG, **overrides}

    def fit(self, X_tr, y_tr, X_val, y_val, cat_cols, X_test=None):
        p = self.p
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        num_cols = [c for c in X_tr.columns if c not in cat_cols]
        X_tr_num = X_tr[num_cols].values.astype(np.float32)
        X_val_num = X_val[num_cols].values.astype(np.float32)
        X_tr_cat = X_tr[cat_cols].values.astype(np.int64)
        X_val_cat = X_val[cat_cols].values.astype(np.int64)
        y_tr_a = np.asarray(y_tr)
        y_va_a = np.asarray(y_val)

        self.preproc_ = NumericalPreprocessor(p["tfms"])
        self.preproc_.fit(X_tr_num)
        X_tr_num = self.preproc_.transform(X_tr_num)
        X_val_num = self.preproc_.transform(X_val_num)

        if cat_cols:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_cols].values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)
        else:
            cat_dims = []
        self.cat_cols_ = cat_cols
        self.num_cols_ = num_cols
        self.cat_dims_ = cat_dims

        classes = np.unique(y_tr_a)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr_a)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

        self.model_ = RealMLPNet(output_dim=len(classes), cat_dims=cat_dims, n_numerical=X_tr_num.shape[1], cfg=p).to(dev)
        groups = _parameter_groups(self.model_, p)
        for g in groups:
            g["lr_base"] = g["lr"]
        opt = torch.optim.AdamW(groups, betas=(p["mom"], p["sq_mom"]))

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr_a, dtype=torch.long, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

        n_ens = p["n_ens"]
        train_bs = p["train_bs"]
        eval_bs = p["eval_bs"]
        epochs = p["epochs"]
        total_steps = epochs * len(y_tr_a)
        order = np.arange(len(y_tr_a))
        best_score = -np.inf
        best_probs = None
        best_epoch = 0

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr_a), train_bs):
                progress = (epoch * len(y_tr_a) + start) / total_steps
                idx = order[start:start + train_bs]
                for g in opt.param_groups:
                    g["lr"] = _apply_schedule(g["lr_base"], progress, p["lr_sched"], p["flat_ratio"])
                opt.zero_grad()
                y_pred = self.model_(Xtn[idx], Xtc[idx])
                ls_val = _apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], p["flat_ratio"])
                drop_val = _apply_schedule(p["dropout"], progress, p["p_drop_sched"], p["flat_ratio"])
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val
                loss = _smooth_ce(ytt[idx].repeat_interleave(n_ens), y_pred.reshape(-1, len(classes)),
                                  ls=ls_val, class_weights=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                opt.step()
            np.random.shuffle(order)

            self.model_.eval()
            with torch.no_grad():
                val_probs = np.concatenate([
                    self.model_(Xvn[s:s + eval_bs], Xvc[s:s + eval_bs]).mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_va_a), eval_bs)
                ], axis=0)
            score = balanced_accuracy_score(y_va_a, np.argmax(val_probs, axis=1))
            if score > best_score:
                best_score = score
                best_probs = val_probs.copy()
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            if p.get("verbosity", 1) >= 2:
                print(f"  epoch {epoch + 1}/{epochs}  bal-acc={score:.5f}  best={best_score:.5f}")

        self.model_.load_state_dict(best_state)
        self.best_score_ = best_score
        self.best_val_probs_ = best_probs
        self.best_epoch_ = best_epoch
        self._dev = dev
        if p.get("verbosity", 1) >= 1:
            print(f"  RealMLP best bal-acc: {best_score:.5f} (epoch {best_epoch})")
        return self

    def predict_proba(self, X):
        eval_bs = self.p["eval_bs"]
        X_num = self.preproc_.transform(X[self.num_cols_].values.astype(np.float32))
        X_cat = X[self.cat_cols_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                self.model_(Xn[s:s + eval_bs], Xc[s:s + eval_bs]).mean(dim=1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)
