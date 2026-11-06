"""A self-contained HTML summary of measured milestone-one results."""

import base64
from html import escape
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


PALETTE = {"random_forest": "#087f82", "mlp": "#d48a34"}
LABELS = {"random_forest": "Random forest", "mlp": "MLP"}


def image_tag(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = base64.b64encode(buffer.getvalue()).decode()
    return '<img alt="Measured experiment chart" src="data:image/png;base64,' + data + '">'


def build_report(audit_path: Path, run_dir: Path) -> Path:
    audit = json.loads(audit_path.read_text())
    result = json.loads((run_dir / "results.json").read_text())
    predictions = pd.read_csv(run_dir / "validation_predictions.csv")
    counts = pd.DataFrame(audit["per_time_step"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(counts.time_step, counts.licit, color="#087f82", label="Known licit")
    ax.bar(counts.time_step, counts.illicit, bottom=counts.licit, color="#d48a34", label="Known illicit")
    for step in [16.5, 24.5, 34.5]:
        ax.axvline(step, color="#81919b", linestyle="--", linewidth=1)
    ax.set(xlabel="Original Elliptic time step", ylabel="Known labels", title="Dataset audit: labelled transactions over time")
    ax.legend(frameon=False, ncols=2)
    timeline = image_tag(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, metrics in result["models"].items():
        precision, recall, _ = precision_recall_curve(predictions.true_illicit, predictions[name + "_score"])
        ax.plot(recall, precision, color=PALETTE[name], label=f"{LABELS[name]} (AP {metrics['average_precision']:.3f})")
    ax.set(xlabel="Illicit recall", ylabel="Illicit precision", title="Chronological validation: precision-recall", xlim=(0, 1), ylim=(0, 1.03))
    ax.legend(frameon=False, loc="lower left")
    pr = image_tag(fig)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name in result["models"]:
        points = []
        for step, group in predictions.groupby("time_step"):
            if group.true_illicit.nunique() < 2:
                points.append((step, np.nan))
            else:
                from sklearn.metrics import average_precision_score
                points.append((step, average_precision_score(group.true_illicit, group[name + "_score"])))
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", color=PALETTE[name], label=LABELS[name])
    ax.set(xlabel="Validation time step", ylabel="Average precision", title="Validation quality over time", ylim=(0, 1.03))
    ax.legend(frameon=False)
    temporal = image_tag(fig)
    rows = []
    for name, m in result["models"].items():
        rows.append(f"<tr><td>{escape(LABELS[name])}</td><td>{m['average_precision']:.4f}</td>"
                    f"<td>{m['precision_illicit']:.4f}</td><td>{m['recall_illicit']:.4f}</td>"
                    f"<td>{m['f1_illicit']:.4f}</td><td>{m['threshold']:.4f}</td></tr>")
    split_rows = ''.join(f"<tr><td>{escape(role.title())}</td><td>{v['time_steps'][0]}-{v['time_steps'][1]}</td>"
                         f"<td>{v['nodes']:,}</td><td>{v['illicit']:,}</td><td>{v['unknown']:,}</td></tr>"
                         for role, v in audit["splits"].items())
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreamGraph | Milestone 1 measured results</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#243744;background:#f3f6f8;margin:0}}main{{max-width:1040px;margin:auto;padding:40px 24px}}
h1,h2{{color:#172f45;line-height:1.2}}h1{{font-size:38px}}.tag{{color:#087f82;font-weight:700;letter-spacing:.06em}}section{{background:white;border:1px solid #d6e1e5;border-radius:12px;padding:25px;margin:22px 0}}img{{display:block;max-width:100%;height:auto;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #d6e1e5}}th{{background:#eaf4f3}}.note{{background:#fcf5e7;border-left:4px solid #d48a34;padding:15px}}code{{font-size:13px;overflow-wrap:anywhere}}small{{color:#5a6b76}}.scroll{{overflow:auto}}
</style><main><div class="tag">PREETESH / FINANCIAL GRAPH LEARNING</div><h1>The first measured baseline</h1>
<p>Elliptic data audit, chronological splits, random forest, and MLP. Generated {escape(result['created_at'])}.</p>
<p class="note"><b>Validation / model selection only.</b> Epochs and thresholds were selected using validation data. Final test steps 35-49 were not scored. One initial seed is not a final research comparison.</p>
<section><h2>Dataset integrity</h2><p>{audit['nodes']:,} transaction nodes; {audit['edges']:,} directed edges; {audit['numeric_features_without_time']} numerical features after excluding time. Unknown labels are -1, excluded from supervised loss and metrics.</p>
<div class="scroll"><table><tr><th>Role</th><th>Steps</th><th>All nodes</th><th>Known illicit</th><th>Unknown</th></tr>{split_rows}</table></div>{timeline}</section>
<section><h2>Baseline measurements</h2><p>Training used known source labels from steps 1-16. Adaptation-prefix labels were not used. The MLP scaler was fitted to source training rows only.</p>
<div class="scroll"><table><tr><th>Model</th><th>AP</th><th>Precision</th><th>Recall</th><th>F1</th><th>Threshold</th></tr>{''.join(rows)}</table></div>
<p><small>Illicit is the positive class. Thresholds maximize validation F1. Average precision (AP) measures ranking and does not depend on the threshold.</small></p>{pr}{temporal}</section>
<section><h2>Implemented and next</h2><p><b>Implemented:</b> reproducible download, ID-based joins, integrity checks, unknown-label masking, time-separated splits, source-only preprocessing, baselines, graph-snapshot export, and this report.</p>
<p><b>Next:</b> GraphSAGE on the same split and features; then DANN/GRL; then Spark snapshot replay. No GNN, DANN, or Spark performance is claimed in this milestone.</p></section>
<section><h2>Reproducibility</h2><p>Seed: {result['seed']}; feature view: {escape(result['feature_view'])}; source training nodes: {result['fit_audit']['source_rows']:,}.</p>
<p>Config SHA-256: <code>{escape(result['config_sha256'])}</code><br>Dataset fingerprint: <code>{escape(result['dataset_fingerprint'])}</code></p>
<p>Data source: the <a href="https://data.pyg.org/datasets/elliptic">PyG-maintained Elliptic mirror</a>, matching the original <a href="https://www.kaggle.com/datasets/ellipticco/elliptic-data-set">Elliptic release</a>.</p></section></main></html>'''
    path = run_dir / "report.html"
    path.write_text(html)
    return path
