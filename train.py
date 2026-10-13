"""
Training pipeline: Rent the Runway "will this rental not fit?" classifier.

Run with:  uv run python main.py train

Steps
1. Load the committed sample, build the binary target (misfit = small or large).
2. Chronological split 70/15/15 by rental date (train = past, test = future).
3. Feature engineering (features.py): parsing, past-only history rates,
   expected-size deviation fitted on train only.
4. Preprocess (impute + scale + one-hot), fitted on train only.
5. Feature-group ablation with scikit-learn on the validation set.
6. Logistic regression three ways on the same matrices:
   scikit-learn, manual PyTorch loop, nn.Module + torch.optim.
7. Pick each model's decision threshold on VALIDATION by business cost,
   then report everything on TEST against naive baselines.
8. Save models + predictions + metrics to artifacts/ for app.py.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import features as F

ART = Path("artifacts")
ART.mkdir(exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Business costs (EUR per rental) — see REPORT.md "Business framing".
COST_FN = 40.0  # missed misfit: customer has nothing to wear for the event -> refund credit + churn risk
COST_FP = 8.0   # unnecessary backup size: extra shipping + dry-cleaning of an unworn item

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
LR_GRID = [0.1, 0.5, 1.0]
EPOCHS = 3000


# ======================================================================= data
def chronological_split(df):
    cut_train, cut_val = df["review_date"].quantile([0.70, 0.85])
    split = np.where(df["review_date"] <= cut_train, "train",
                     np.where(df["review_date"] <= cut_val, "val", "test"))
    return pd.Series(split, index=df.index)


def build_dataset():
    df = F.parse_row_features(F.make_target(F.load_raw()))
    df["split"] = chronological_split(df)
    train = df[df["split"] == "train"]
    prior = {"small": train["is_small"].mean(), "large": train["is_large"].mean()}

    df = F.add_history_rates(df, "item_id", "item", prior)
    df = F.add_history_rates(df, "user_id", "user", prior)
    size_model = F.ExpectedSize().fit(df[df["split"] == "train"])
    df = size_model.transform(df)
    return df, prior, size_model


def make_preprocessor(numeric):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), F.CATEGORICAL),
    ])


# ======================================================================= models
def penalty_lambda(C, n):
    """sklearn minimises  C * sum(logloss) + 0.5*||w||^2.
    Dividing by C*n gives the equivalent per-sample objective
        mean(logloss) + (1 / (2*C*n)) * ||w||^2,
    i.e. an L2 term (lambda/2)*||w||^2 with lambda = 1/(C*n).
    Using the same objective in PyTorch is what makes the three comparable."""
    return 1.0 / (C * n)


def train_manual(X, y, lam, lr, epochs=EPOCHS):
    """Session-5 style: raw tensors, autograd, hand-written update."""
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for _ in range(epochs):
        z = X @ w + b
        p = torch.sigmoid(z).clamp(1e-7, 1 - 1e-7)
        bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        loss = bce + 0.5 * lam * (w ** 2).sum()  # bias is NOT penalised (same as sklearn)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        losses.append(loss.item())
    return {"w": w.detach(), "b": b.detach()}, losses


class LogReg(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = torch.nn.Linear(d, 1)
        # Zero init, same as the manual loop, so differences come from the
        # optimiser plumbing, not from the starting point.
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits; sigmoid lives in the loss


def train_module(X, y, lam, lr, epochs=EPOCHS):
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    model = LogReg(X.shape[1])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # weight_decay=lam adds lam*w to the gradient == gradient of (lam/2)||w||^2.
    # Put the bias in its own group with no decay, to match sklearn.
    opt = torch.optim.SGD([
        {"params": [model.linear.weight], "weight_decay": lam},
        {"params": [model.linear.bias], "weight_decay": 0.0},
    ], lr=lr)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        bce = loss_fn(model(X), y)
        bce.backward()
        opt.step()
        losses.append(bce.item() + 0.5 * lam * (model.linear.weight ** 2).sum().item())
    return model, losses


def predict_manual(params, X):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X, dtype=torch.float32) @ params["w"] + params["b"]).numpy()


def predict_module(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# ======================================================================= evaluation
def expected_cost(y, p, t):
    pred = p >= t
    fn = ((~pred) & (y == 1)).sum()
    fp = (pred & (y == 0)).sum()
    return (COST_FN * fn + COST_FP * fp) / len(y)


def best_threshold(y, p):
    grid = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y, p, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def scores(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    out = {
        "threshold": round(t, 3),
        "cost_per_rental_eur": round(expected_cost(y, p, t), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "flag_rate": round(pred.mean(), 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if len(np.unique(p)) > 1:
        out.update(roc_auc=round(roc_auc_score(y, p), 4), pr_auc=round(average_precision_score(y, p), 4))
    else:
        out.update(roc_auc=0.5, pr_auc=round(y.mean(), 4))
    out.update(log_loss=round(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]), 4),
               brier=round(brier_score_loss(y, p), 4))
    return out


# ======================================================================= main
def main():
    df, prior, size_model = build_dataset()
    tr, va, te = (df[df["split"] == s] for s in ["train", "val", "test"])
    y_tr, y_va, y_te = (d["misfit"].to_numpy() for d in (tr, va, te))
    print(f"rows  train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    for name, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {name:5s} {d['review_date'].min().date()} .. {d['review_date'].max().date()}"
              f"  misfit rate={d['misfit'].mean():.3f}")

    # ---------------- ablation: add feature groups one at a time (sklearn, C=1)
    ablation, used = [], []
    for group, cols in F.FEATURE_GROUPS.items():
        used = used + cols
        pre = make_preprocessor(used).fit(tr)
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(pre.transform(tr), y_tr)
        p = clf.predict_proba(pre.transform(va))[:, 1]
        ablation.append({"added_group": group, "n_numeric": len(used),
                         "val_roc_auc": round(roc_auc_score(y_va, p), 4),
                         "val_log_loss": round(log_loss(y_va, p), 4),
                         "val_best_cost": round(expected_cost(y_va, p, best_threshold(y_va, p)), 3)})
    print("\nAblation (cumulative, validation set):")
    print(pd.DataFrame(ablation).to_string(index=False))

    # ---------------- final design matrices (all feature groups + categoricals)
    pre = make_preprocessor(F.NUMERIC_FEATURES).fit(tr)
    X_tr, X_va, X_te = (pre.transform(d) for d in (tr, va, te))
    feat_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    n = len(X_tr)
    print(f"\nDesign matrix: {X_tr.shape[1]} columns")

    # ---------------- 1) scikit-learn, tune C on val log loss
    c_results = {}
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=5000).fit(X_tr, y_tr)
        c_results[C] = log_loss(y_va, m.predict_proba(X_va)[:, 1])
    best_C = min(c_results, key=c_results.get)
    sk = LogisticRegression(C=best_C, max_iter=5000).fit(X_tr, y_tr)
    lam = penalty_lambda(best_C, n)
    print(f"sklearn: val log loss by C {({k: round(v, 4) for k, v in c_results.items()})} -> C={best_C}, lambda={lam:.2e}")

    # ---------------- 2) manual PyTorch, tune lr on val log loss
    lr_results = {}
    for lr in LR_GRID:
        params, losses = train_manual(X_tr, y_tr, lam, lr)
        lr_results[lr] = (log_loss(y_va, predict_manual(params, X_va)), params, losses)
    best_lr_manual = min(lr_results, key=lambda k: lr_results[k][0])
    manual_params, manual_losses = lr_results[best_lr_manual][1:]
    print(f"manual : val log loss by lr {({k: round(v[0], 4) for k, v in lr_results.items()})} -> lr={best_lr_manual}")

    # ---------------- 3) nn.Module + torch.optim
    lr_results = {}
    for lr in LR_GRID:
        model, losses = train_module(X_tr, y_tr, lam, lr)
        lr_results[lr] = (log_loss(y_va, predict_module(model, X_va)), model, losses)
    best_lr_module = min(lr_results, key=lambda k: lr_results[k][0])
    module, module_losses = lr_results[best_lr_module][1:]
    print(f"nn.Module: val log loss by lr {({k: round(v[0], 4) for k, v in lr_results.items()})} -> lr={best_lr_module}")

    # ---------------- predictions
    probs = {
        "sklearn": (sk.predict_proba(X_va)[:, 1], sk.predict_proba(X_te)[:, 1]),
        "manual_torch": (predict_manual(manual_params, X_va), predict_manual(manual_params, X_te)),
        "nn_module": (predict_module(module, X_va), predict_module(module, X_te)),
        # Naive baseline: everyone gets the training misfit rate.
        "baseline_prior": (np.full(len(va), y_tr.mean()), np.full(len(te), y_tr.mean())),
    }

    results = {}
    for name, (p_va, p_te) in probs.items():
        t = best_threshold(y_va, p_va)  # chosen on VALIDATION, never on test
        results[name] = scores(y_te, p_te, t)
    # Two policy baselines with no model at all.
    results["policy_never_backup"] = scores(y_te, np.zeros(len(te)), 0.5)
    results["policy_always_backup"] = scores(y_te, np.ones(len(te)), 0.5)
    # Theory check: with calibrated probabilities the cost-optimal threshold is FP/(FP+FN).
    theory_t = COST_FP / (COST_FP + COST_FN)

    table = pd.DataFrame(results).T
    print(f"\nTEST results (thresholds picked on val; theoretical optimum = {theory_t:.3f}):")
    print(table[["threshold", "cost_per_rental_eur", "roc_auc", "pr_auc", "log_loss", "brier",
                 "precision", "recall", "flag_rate"]].to_string())

    # ---------------- do the three agree?
    w_sk = np.r_[sk.intercept_, sk.coef_.ravel()]
    w_man = np.r_[manual_params["b"].numpy(), manual_params["w"].numpy()]
    w_mod = np.r_[module.linear.bias.detach().numpy(), module.linear.weight.detach().numpy().ravel()]
    p_sk, p_man, p_mod = probs["sklearn"][1], probs["manual_torch"][1], probs["nn_module"][1]
    t_sk = results["sklearn"]["threshold"]
    agreement = {
        "max_abs_coef_diff_manual_vs_sklearn": float(np.abs(w_man - w_sk).max()),
        "max_abs_coef_diff_module_vs_sklearn": float(np.abs(w_mod - w_sk).max()),
        "max_abs_coef_diff_manual_vs_module": float(np.abs(w_man - w_mod).max()),
        "max_abs_prob_diff_manual_vs_sklearn": float(np.abs(p_man - p_sk).max()),
        "max_abs_prob_diff_module_vs_sklearn": float(np.abs(p_mod - p_sk).max()),
        "decision_agreement_at_sklearn_threshold": {
            "manual_torch": float(((p_man >= t_sk) == (p_sk >= t_sk)).mean()),
            "nn_module": float(((p_mod >= t_sk) == (p_sk >= t_sk)).mean()),
        },
        # Same objective for all three: mean BCE on train + (lambda/2)*||w||^2.
        # If sklearn's value is lower, the PyTorch runs simply haven't fully converged.
        "final_train_objective": {
            "sklearn": float(log_loss(y_tr, sk.predict_proba(X_tr)[:, 1]) + 0.5 * lam * (sk.coef_ ** 2).sum()),
            "manual_torch": manual_losses[-1], "nn_module": module_losses[-1]},
    }
    print("\nAgreement:", json.dumps(agreement, indent=2))

    coefs = pd.DataFrame({"feature": ["(intercept)"] + feat_names,
                          "sklearn": w_sk, "manual_torch": w_man, "nn_module": w_mod})
    print("\nLargest sklearn coefficients (standardised features):")
    print(coefs.reindex(coefs["sklearn"].abs().sort_values(ascending=False).index).head(12).to_string(index=False))

    # ---------------- save artifacts (the app loads these, never retrains)
    with open(ART / "preprocessing.pkl", "wb") as f:
        pickle.dump({"preprocessor": pre, "size_model": size_model, "prior": prior,
                     "item_snapshot": F.history_snapshot(df, "item_id", prior)}, f)
    with open(ART / "sklearn_logreg.pkl", "wb") as f:
        pickle.dump(sk, f)
    torch.save(manual_params, ART / "manual_torch.pt")
    torch.save(module.state_dict(), ART / "nn_module.pt")

    # Engineered features for every row (for the distribution views).
    info = ["review_date", "split", "item_id", "fit", "misfit", "size_dev", "item_misfit_rate"] + F.CATEGORICAL
    df[info + F.NUMERIC_FEATURES].to_csv(ART / "features_all.csv", index=False)

    out = pd.concat([va, te])[["review_date", "split", "item_id", "category", "size", "fit", "misfit"]]
    for name, (p_va, p_te) in probs.items():
        out[f"p_{name}"] = np.r_[p_va, p_te]
    out.to_csv(ART / "predictions.csv", index=False)
    coefs.to_csv(ART / "coefficients.csv", index=False)

    with open(ART / "metrics.json", "w") as f:
        json.dump({
            "costs": {"fn": COST_FN, "fp": COST_FP, "theoretical_threshold": theory_t},
            "split": {s: {"rows": int((df["split"] == s).sum()),
                          "start": str(df.loc[df["split"] == s, "review_date"].min().date()),
                          "end": str(df.loc[df["split"] == s, "review_date"].max().date()),
                          "misfit_rate": float(df.loc[df["split"] == s, "misfit"].mean())}
                      for s in ["train", "val", "test"]},
            "hyperparameters": {"C": best_C, "lambda": lam, "lr_manual": best_lr_manual,
                                "lr_module": best_lr_module, "epochs": EPOCHS,
                                "val_log_loss_by_C": c_results},
            "ablation": ablation,
            "test_results": results,
            "agreement": agreement,
            "loss_curves": {"manual_torch": manual_losses[::10], "nn_module": module_losses[::10]},
            "n_features": int(X_tr.shape[1]),
        }, f, indent=2, default=float)
    print(f"\nSaved artifacts to {ART.resolve()}")


if __name__ == "__main__":
    main()
