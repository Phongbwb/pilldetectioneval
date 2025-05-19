import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fine_tune import fine_tune
from copy import deepcopy
from itertools import product
import pandas as pd

HYPERPARAM_GRID = {
    "yolov11": {
        "lr": [0.001, 0.0005],
        "batch_size": [16, 32],
        "img_size": [416, 640]
    },
    "faster_rcnn": {
        "lr": [0.01, 0.005],
        "batch_size": [8, 16],
        "backbone": ["resnet50", "resnet101"]
    },
    "retinanet": {
        "lr": [0.01, 0.001],
        "batch_size": [8, 16],
        "focal_alpha": [0.25, 0.5]
    },
    "ssd": {
        "lr": [0.001, 0.0001],
        "batch_size": [16, 32],
        "img_size": [300, 512]
    },
    "rt_detr": {
        "lr": [0.0001, 5e-5],
        "batch_size": [8, 16],
        "num_queries": [100, 300]
    }
}

def grid_search(model_name, param_grid):
    keys = list(param_grid.keys())
    best_map = -1
    best_config = None
    history = []

    for values in product(*[param_grid[k] for k in keys]):
        config = dict(zip(keys, values))
        print(f" Testing {model_name} config: {config}")
        mAP = fine_tune(model_name, config)
        print(f" mAP: {mAP}")
        record = deepcopy(config)
        record["mAP"] = mAP
        history.append(record)

        if mAP > best_map:
            best_map = mAP
            best_config = deepcopy(config)

    os.makedirs("configs", exist_ok=True)
    with open(f"configs/{model_name}.yaml", "w") as f:
        yaml.dump(best_config, f)

    # Save and plot results
    df = pd.DataFrame(history)
    df.to_csv(f"configs/{model_name}_search_results.csv", index=False)
    plot_correlation(model_name, df)

    print(f" Best config for {model_name}: {best_config}, mAP: {best_map}")

def plot_correlation(model_name, df):
    os.makedirs("plots", exist_ok=True)
    for col in df.columns:
        if col == "mAP":
            continue
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df["mAP"])
        plt.title(f"{model_name.upper()} - {col} vs mAP")
        plt.xlabel(col)
        plt.ylabel("mAP")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_{col}_vs_mAP.png")
        plt.close()

def main():
    for model_name, grid in HYPERPARAM_GRID.items():
        print(f"\n Starting hyperparameter search for {model_name.upper()}")
        grid_search(model_name, grid)

if __name__ == "__main__":
    main()