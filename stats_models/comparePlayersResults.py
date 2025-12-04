import os
import pandas as pd

REAL_PATH = "../database/final/players_teams.csv"
PRED_PATH = "players_predictions"

real_df = pd.read_csv(REAL_PATH)
real_df["playerID"] = real_df["playerID"].astype(str)
real_df["year"] = real_df["year"].astype(int)
real_df["tmID"] = real_df["tmID"].astype(str)

OUTPUT_DIR = "players_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compare_file(pred_file):
    pred_df = pd.read_csv(os.path.join(PRED_PATH, pred_file))
    pred_df["playerID"] = pred_df["playerID"].astype(str)
    pred_df["year"] = pred_df["year"].astype(int)
    pred_df["tmID"] = pred_df["tmID"].astype(str)

    merged = pred_df.merge(
        real_df,
        on=["playerID", "year", "tmID"],
        suffixes=("_pred", "_real")
    )

    cols = [c.replace("_pred", "") for c in merged.columns if c.endswith("_pred")]

    result = merged[["playerID", "year", "tmID"]].copy()

    for col in cols:
        pred_col = col + "_pred"
        real_col = col + "_real"

        result[col + "_pct_error"] = (
            ((merged[pred_col] - merged[real_col]).abs()
            / merged[real_col].replace(0, 1).abs()) * 100
        ).round(1).astype(str) + "%"


    year = merged["year"].iloc[0]
    out_path = os.path.join(OUTPUT_DIR, f"comparison_year_{year}.csv")
    result.to_csv(out_path, index=False)
    print(f"{out_path}")


for fname in os.listdir(PRED_PATH):
    if fname.endswith(".csv"):
        compare_file(fname)
