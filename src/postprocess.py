import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_submission(test_df, preds, output_dir: Path):
    sub = pd.DataFrame({'id': range(len(preds)), 'target': preds})
    out_path = output_dir / "sample_submission.csv"
    sub.to_csv(out_path, index=False)
    return out_path

def save_importances(model, features, output_dir: Path):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:5]
    top5 = {f: float(i) for f, i in pairs}
    path = output_dir / "feature_importances_top5.json"
    with open(path, "w") as f:
        json.dump(top5, f, indent=2)
    return path

def save_density_plot(preds, output_dir: Path):
    plt.figure(figsize=(6,4))
    plt.hist(preds, bins=50, density=True)
    plt.title("Score Density")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    path = output_dir / "score_density.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
