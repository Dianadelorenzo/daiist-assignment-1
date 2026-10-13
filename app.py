"""
Gradio dashboard for the Rent the Runway misfit model.

Run with:  uv run python main.py app      (after `main.py train`)

Everything here is LOADED from artifacts/ written by train.py. Nothing is fitted
or retrained: the "Try a rental" tab only calls .transform / forward passes.
"""

import json
import os
import pickle
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

import features as F
from train import LogReg

ART = Path("artifacts")
if not (ART / "metrics.json").exists():
    raise SystemExit("artifacts/ not found - run `uv run python main.py train` first.")

METRICS = json.loads((ART / "metrics.json").read_text())
PRED = pd.read_csv(ART / "predictions.csv")
FEATS = pd.read_csv(ART / "features_all.csv", parse_dates=["review_date"])
COEFS = pd.read_csv(ART / "coefficients.csv")
with open(ART / "preprocessing.pkl", "rb") as f:
    PREP = pickle.load(f)
with open(ART / "sklearn_logreg.pkl", "rb") as f:
    SK = pickle.load(f)
MANUAL = torch.load(ART / "manual_torch.pt")
MODULE = LogReg(METRICS["n_features"])
MODULE.load_state_dict(torch.load(ART / "nn_module.pt"))
MODULE.eval()

MODELS = {"sklearn": "scikit-learn", "manual_torch": "Manual PyTorch loop", "nn_module": "nn.Module + optim"}
COLORS = {"sklearn": "#1f77b4", "manual_torch": "#ff7f0e", "nn_module": "#2ca02c", "baseline_prior": "#7f7f7f"}
COST_FN, COST_FP = METRICS["costs"]["fn"], METRICS["costs"]["fp"]


# ============================================================ comparison tab
def results_table():
    rows = []
    labels = {**MODELS, "baseline_prior": "Baseline: training misfit rate",
              "policy_never_backup": "Policy: never send backup", "policy_always_backup": "Policy: always send backup"}
    for key, r in METRICS["test_results"].items():
        rows.append({"model": labels[key], "threshold (from val)": r["threshold"],
                     "cost / rental (EUR)": r["cost_per_rental_eur"], "ROC AUC": r["roc_auc"],
                     "PR AUC": r["pr_auc"], "log loss": r["log_loss"], "Brier": r["brier"],
                     "precision": r["precision"], "recall": r["recall"], "% flagged": r["flag_rate"]})
    return pd.DataFrame(rows)


def calibration_plot(split):
    """Prediction vs actual for a classifier: bin predicted probabilities and
    compare each bin's mean prediction with the observed misfit rate."""
    d = PRED[PRED["split"] == split]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.8], y=[0, 0.8], mode="lines", name="perfect calibration",
                             line=dict(dash="dash", color="black")))
    for key, label in MODELS.items():
        bins = pd.qcut(d[f"p_{key}"], 10, duplicates="drop")
        g = d.groupby(bins, observed=True).agg(pred=(f"p_{key}", "mean"), actual=("misfit", "mean"), n=("misfit", "size"))
        fig.add_trace(go.Scatter(x=g["pred"], y=g["actual"], mode="lines+markers", name=label,
                                 line=dict(color=COLORS[key]), customdata=g["n"],
                                 hovertemplate="predicted %{x:.3f}<br>actual %{y:.3f}<br>n=%{customdata}"))
    base = d["p_baseline_prior"].iloc[0]
    fig.add_trace(go.Scatter(x=[base], y=[d["misfit"].mean()], mode="markers", name="naive baseline",
                             marker=dict(size=14, symbol="x", color=COLORS["baseline_prior"])))
    fig.update_layout(title=f"Predicted vs actual misfit rate (deciles, {split} set)",
                      xaxis_title="mean predicted P(misfit)", yaxis_title="observed misfit rate", height=450)
    return fig


def agreement_scatter(split):
    d = PRED[PRED["split"] == split]
    fig = go.Figure()
    for key in ["manual_torch", "nn_module"]:
        fig.add_trace(go.Scattergl(x=d["p_sklearn"], y=d[f"p_{key}"], mode="markers", name=MODELS[key],
                                   marker=dict(size=4, opacity=0.5, color=COLORS[key])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="y = x", line=dict(dash="dash", color="black")))
    fig.update_layout(title="Per-rental probability: PyTorch vs scikit-learn", height=450,
                      xaxis_title="scikit-learn P(misfit)", yaxis_title="PyTorch P(misfit)")
    return fig


def prob_histogram(split, model):
    d = PRED[PRED["split"] == split].assign(actual=lambda x: x["fit"])
    return px.histogram(d, x=f"p_{model}", color="actual", barmode="overlay", nbins=50, opacity=0.6,
                        title=f"{MODELS[model]}: predicted P(misfit) by actual outcome ({split})",
                        labels={f"p_{model}": "predicted P(misfit)"}).update_layout(height=400)


def coefficient_plot():
    c = COEFS[COEFS["feature"] != "(intercept)"].copy()
    c = c.reindex(c["sklearn"].abs().sort_values().index).tail(20)
    long = c.melt(id_vars="feature", var_name="model", value_name="coefficient")
    long["model"] = long["model"].map(MODELS)
    return px.bar(long, x="coefficient", y="feature", color="model", barmode="group", orientation="h",
                  title="Top-20 coefficients (standardised features) - all three models",
                  height=650, color_discrete_sequence=list(COLORS.values()))


def loss_plot():
    fig = go.Figure()
    for key in ["manual_torch", "nn_module"]:
        curve = METRICS["loss_curves"][key]
        fig.add_trace(go.Scatter(x=np.arange(len(curve)) * 10, y=curve, name=MODELS[key], line=dict(color=COLORS[key])))
    sk_obj = METRICS["agreement"]["final_train_objective"].get("sklearn")
    if sk_obj is not None:
        fig.add_hline(y=sk_obj, line_dash="dash", annotation_text="scikit-learn optimum")
    fig.update_layout(title="Training objective (mean BCE + L2) per epoch", xaxis_title="epoch",
                      yaxis_title="objective", yaxis_type="log", height=400)
    return fig


def agreement_md():
    a, h = METRICS["agreement"], METRICS["hyperparameters"]
    return (
        f"**Same split, same objective.** C = {h['C']} (chosen on val log loss) -> "
        f"lambda = 1/(C*n) = {h['lambda']:.2e}; PyTorch lr: manual {h['lr_manual']}, "
        f"nn.Module {h['lr_module']}; {h['epochs']} full-batch epochs.\n\n"
        f"| check | value |\n|---|---|\n"
        f"| max abs coefficient diff, manual vs sklearn | {a['max_abs_coef_diff_manual_vs_sklearn']:.4f} |\n"
        f"| max abs coefficient diff, nn.Module vs sklearn | {a['max_abs_coef_diff_module_vs_sklearn']:.4f} |\n"
        f"| max abs coefficient diff, manual vs nn.Module | {a['max_abs_coef_diff_manual_vs_module']:.1e} |\n"
        f"| max abs P(misfit) diff on test, manual vs sklearn | {a['max_abs_prob_diff_manual_vs_sklearn']:.4f} |\n"
        f"| same decision as sklearn on test (manual / module) | "
        f"{a['decision_agreement_at_sklearn_threshold']['manual_torch']:.2%} / "
        f"{a['decision_agreement_at_sklearn_threshold']['nn_module']:.2%} |\n"
    )


# ============================================================ threshold tab
def threshold_view(model, split, threshold, cost_fn, cost_fp):
    d = PRED[PRED["split"] == split]
    y, p = d["misfit"].to_numpy(), d[f"p_{model}"].to_numpy()
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["predicted: fits", "predicted: misfit -> backup"],
        y=["actual: fit", "actual: misfit"], text=[[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix - {MODELS.get(model, model)}, threshold {threshold:.2f} ({split})",
                     height=380, yaxis_autorange="reversed")

    grid = np.linspace(0.01, 0.99, 99)
    costs = [(cost_fn * ((p < t) & (y == 1)).sum() + cost_fp * ((p >= t) & (y == 0)).sum()) / len(y) for t in grid]
    curve = go.Figure(go.Scatter(x=grid, y=costs, name="model", line=dict(color=COLORS.get(model))))
    always = cost_fp * (y == 0).mean()
    never = cost_fn * (y == 1).mean()
    curve.add_hline(y=always, line_dash="dot", annotation_text=f"always backup EUR {always:.2f}")
    curve.add_hline(y=never, line_dash="dot", annotation_text=f"never backup EUR {never:.2f}")
    curve.add_vline(x=threshold, line_color="red")
    curve.add_vline(x=cost_fp / (cost_fp + cost_fn), line_dash="dash", line_color="gray",
                    annotation_text="FP/(FP+FN)")
    curve.update_layout(title="Expected cost per rental vs threshold", xaxis_title="threshold",
                        yaxis_title="EUR / rental", height=380)

    cost = (cost_fn * fn + cost_fp * fp) / len(y)
    best = min(always, never)
    md = (f"### Expected cost: **EUR {cost:.2f} per rental** "
          f"(EUR {cost * 1000:,.0f} per 1,000 rentals)\n"
          f"Best no-model policy: EUR {best:.2f} -> saving **EUR {(best - cost) * 1000:,.0f} per 1,000 rentals**.  \n"
          f"Backups sent: {pred.mean():.1%} of rentals | misfits caught (recall): {tp / max(tp + fn, 1):.1%} | "
          f"backups actually needed (precision): {tp / max(tp + fp, 1):.1%}")
    return cm, curve, md


# ============================================================ distributions tab
NUMERIC_CHOICES = [c for c in F.NUMERIC_FEATURES if c in FEATS.columns]


def feature_distribution(feature, split):
    d = FEATS if split == "all" else FEATS[FEATS["split"] == split]
    return px.histogram(d, x=feature, color="fit", barmode="overlay", nbins=50, opacity=0.6,
                        histnorm="probability density",
                        title=f"{feature} by fit outcome ({split}, density-normalised)").update_layout(height=420)


def target_distribution():
    t = FEATS.groupby(["split", "fit"]).size().reset_index(name="n")
    t["share"] = t["n"] / t.groupby("split")["n"].transform("sum")
    t["split"] = pd.Categorical(t["split"], ["train", "val", "test"])
    return px.bar(t.sort_values("split"), x="split", y="share", color="fit", text_auto=".1%",
                  title="Target distribution by chronological split").update_layout(height=400)


def misfit_over_time():
    m = FEATS.set_index("review_date").resample("ME")["misfit"].agg(["mean", "size"]).reset_index()
    fig = px.line(m, x="review_date", y="mean", markers=True, title="Monthly misfit rate (split boundaries dashed)",
                  labels={"mean": "misfit rate", "review_date": "month"})
    for s in ["val", "test"]:
        fig.add_vline(x=pd.Timestamp(METRICS["split"][s]["start"]).timestamp() * 1000, line_dash="dash")
    return fig.update_layout(height=380)


def ablation_table():
    return pd.DataFrame(METRICS["ablation"])


# ============================================================ what-if tab
ITEMS = (FEATS.groupby("item_id").agg(category=("category", "first"), n=("misfit", "size"))
         .sort_values("n", ascending=False))
ITEM_CHOICES = [f"{i} ({r.category}, {r.n} rentals)" for i, r in ITEMS.iterrows()]


def predict_one(item_choice, size, height_ft, height_in, weight, bust, body_type, rented_for, age):
    item_id = int(item_choice.split(" ")[0])
    raw = pd.DataFrame([{
        "fit": "fit", "user_id": -1, "bust_size": bust or None, "item_id": item_id,
        "weight": f"{weight}lbs" if weight else None, "rented_for": rented_for,
        "body_type": None if body_type == "unknown" else body_type, "category": ITEMS.loc[item_id, "category"],
        "height": f"{int(height_ft)}' {int(height_in)}\"", "size": size, "age": age,
        "review_date": pd.Timestamp.today(),
    }])
    df = F.parse_row_features(raw)
    prior, snap = PREP["prior"], PREP["item_snapshot"]
    s = snap.loc[item_id]
    df["item_small_rate"], df["item_large_rate"] = s["small_rate"], s["large_rate"]
    df["item_log_past_n"] = np.log1p(s["n"])
    # New customer: no rental history -> prior rate, zero past rentals.
    df["user_misfit_rate"], df["user_log_past_n"] = prior["small"] + prior["large"], 0.0
    df = PREP["size_model"].transform(df)
    X = PREP["preprocessor"].transform(df)

    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        p = {"sklearn": SK.predict_proba(X)[0, 1],
             "manual_torch": torch.sigmoid(Xt @ MANUAL["w"] + MANUAL["b"]).item(),
             "nn_module": torch.sigmoid(MODULE(Xt)).item()}
    t = METRICS["test_results"]["sklearn"]["threshold"]
    exp = PREP["size_model"].model.predict(df[F.BODY_COLS].fillna(PREP["size_model"].fill))[0]
    lines = [f"| model | P(misfit) | decision @ {t:.2f} |", "|---|---|---|"]
    for k, v in p.items():
        lines.append(f"| {MODELS[k]} | {v:.3f} | {'ship backup size' if v >= t else 'no backup'} |")
    lines.append(f"\nExpected size for these measurements: **{exp:.1f}** (renting {size}, "
                 f"deviation {df['size_dev'].iloc[0]:+.1f}). Item history: {int(s['n'])} rentals, "
                 f"{s['small_rate']:.1%} small / {s['large_rate']:.1%} large (smoothed).")
    return "\n".join(lines)


# ============================================================ layout
with gr.Blocks(title="Rent the Runway - misfit risk") as demo:
    gr.Markdown("# Will this rental fit? - backup-size decision dashboard\n"
                "Logistic regression trained three ways on a chronological split. "
                f"Train {METRICS['split']['train']['start']} to {METRICS['split']['train']['end']}, "
                f"val to {METRICS['split']['val']['end']}, test to {METRICS['split']['test']['end']}. "
                "All models are loaded from `artifacts/`; nothing is retrained here.")

    with gr.Tab("1 - Model comparison"):
        gr.Markdown("### Test-set results (thresholds chosen on validation by business cost)")
        gr.Dataframe(results_table(), interactive=False)
        gr.Markdown(agreement_md())
        split_c = gr.Radio(["test", "val"], value="test", label="Evaluation set")
        with gr.Row():
            cal = gr.Plot(calibration_plot("test"))
            agr = gr.Plot(agreement_scatter("test"))
        model_h = gr.Dropdown(list(MODELS), value="sklearn", label="Model for histogram")
        hist = gr.Plot(prob_histogram("test", "sklearn"))
        split_c.change(calibration_plot, split_c, cal).then(agreement_scatter, split_c, agr) \
            .then(prob_histogram, [split_c, model_h], hist)
        model_h.change(prob_histogram, [split_c, model_h], hist)
        with gr.Row():
            gr.Plot(coefficient_plot())
            gr.Plot(loss_plot())

    with gr.Tab("2 - Threshold & business cost"):
        with gr.Row():
            model_t = gr.Dropdown(list(MODELS) + ["baseline_prior"], value="sklearn", label="Model")
            split_t = gr.Radio(["test", "val"], value="test", label="Set")
            thr = gr.Slider(0.01, 0.99, value=METRICS["test_results"]["sklearn"]["threshold"], step=0.01,
                            label="Decision threshold: ship a backup size if P(misfit) >= threshold")
        with gr.Row():
            cfn = gr.Number(COST_FN, label="Cost of a missed misfit (FN), EUR")
            cfp = gr.Number(COST_FP, label="Cost of an unneeded backup (FP), EUR")
        cost_md = gr.Markdown()
        with gr.Row():
            cm_plot, cost_plot = gr.Plot(), gr.Plot()
        inputs = [model_t, split_t, thr, cfn, cfp]
        for comp in inputs:
            comp.change(threshold_view, inputs, [cm_plot, cost_plot, cost_md])
        demo.load(threshold_view, inputs, [cm_plot, cost_plot, cost_md])

    with gr.Tab("3 - Data & features"):
        with gr.Row():
            gr.Plot(target_distribution())
            gr.Plot(misfit_over_time())
        with gr.Row():
            feat = gr.Dropdown(NUMERIC_CHOICES, value="size_dev_up", label="Feature")
            split_f = gr.Radio(["all", "train", "val", "test"], value="all", label="Split")
        fplot = gr.Plot(feature_distribution("size_dev_up", "all"))
        feat.change(feature_distribution, [feat, split_f], fplot)
        split_f.change(feature_distribution, [feat, split_f], fplot)
        gr.Markdown("### Feature-group ablation (scikit-learn, C=1, validation set, groups added cumulatively)")
        gr.Dataframe(ablation_table(), interactive=False)

    with gr.Tab("4 - Try a rental"):
        gr.Markdown("Score a hypothetical new customer with the saved models (a new customer has no history).")
        with gr.Row():
            item = gr.Dropdown(ITEM_CHOICES, value=ITEM_CHOICES[0], label="Item")
            size = gr.Slider(0, 58, value=8, step=1, label="Size rented (RTR scale)")
            rented_for = gr.Dropdown(sorted(FEATS["rented_for"].dropna().unique()), value="wedding", label="Rented for")
        with gr.Row():
            hft = gr.Number(5, label="Height (ft)", precision=0)
            hin = gr.Number(5, label="Height (in)", precision=0)
            weight = gr.Number(135, label="Weight (lbs, 0 = not given)", precision=0)
            bust = gr.Textbox("34b", label="Bust size (e.g. 34b, blank = not given)")
            body = gr.Dropdown(sorted(FEATS["body_type"].unique()), value="hourglass", label="Body type")
            age = gr.Number(32, label="Age")
        btn = gr.Button("Predict", variant="primary")
        out = gr.Markdown()
        btn.click(predict_one, [item, size, hft, hin, weight, bust, body, rented_for, age], out)


if __name__ == "__main__":
    # On Google Colab localhost isn't reachable from your browser, so ask Gradio
    # for a temporary public *.gradio.live link instead.
    on_colab = "COLAB_RELEASE_TAG" in os.environ
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=not on_colab, share=on_colab)
