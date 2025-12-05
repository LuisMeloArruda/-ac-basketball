import os
import re
import pandas as pd
from sklearn.metrics import confusion_matrix

def extract_year(filename):
    m = re.search(r"year_(\d+)", filename)
    return int(m.group(1)) if m else 999

def extract_conf(filename):
    m = re.search(r"rank_predictions_(\w+)_year", filename)
    return m.group(1) if m else "UNKNOWN"

def evaluate_rank_predictions(folder="teams_rank_predictions"):

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    groups = {}
    for f in files:
        conf = extract_conf(f)
        if conf not in groups:
            groups[conf] = []
        groups[conf].append(f)

    for conf, conf_files in groups.items():

        print(f"\n\n===== CONFERÊNCIA {conf} =====\n")
        print(f"{'FILE':<35} | {'ACERTOS':<10} | {'TOTAL':<10} | {'ACCURACY':<10}")
        print("-" * 75)

        conf_files = sorted(conf_files, key=lambda f: extract_year(f))

        for f in conf_files:
            path = os.path.join(folder, f)
            df = pd.read_csv(path)

            correct = (df["rank"] == df["rank_pred"]).sum()
            total = len(df)
            accuracy = correct / total if total > 0 else 0

            print(
                f"{f:<35} | "
                f"{correct:<10} | "
                f"{total:<10} | "
                f"{accuracy:.0%}"
            )
        print("")


if __name__ == "__main__":
    evaluate_rank_predictions()
