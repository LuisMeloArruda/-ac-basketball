import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# predicted offense + predicted defense
pred = pd.read_csv("./outputs/predicted_team_stats.csv")

# real stats
real = pd.read_csv("../database/final/teams.csv")

# merge predicted + real
merged = pred.merge(
    real,
    on=["tmID", "year"],
    suffixes=("_pred", "_real")
)

# offensive stats
off_cols = [
    "o_fgm", "o_fga", "o_ftm", "o_fta", "o_3pm",
    "o_oreb", "o_dreb", "o_reb",
    "o_asts", "o_pf", "o_stl", "o_to", "o_blk",
    "o_pts"
]

# defensive stats (auto-detected)
def_cols = [c for c in real.columns if c.startswith("d_")]

# full list
cols = off_cols + def_cols

diff = pd.DataFrame()
diff["tmID"] = merged["tmID"]
diff["year"] = merged["year"]

# compute % error
for c in cols:
    real_vals = merged[c + "_real"]
    pred_vals = merged[c + "_pred"]

    abs_err = (real_vals - pred_vals).abs()

    percent_err = np.where(
        real_vals != 0,
        (abs_err / real_vals.abs()) * 100,
        0
    )

    diff[c + "_err_percent"] = percent_err.round(2)

# save output
diff.to_csv("outputs/team_stats_diff_percent.csv", index=False)

# model-wide metrics
y_true = merged[[c + "_real" for c in cols]].values
y_pred = merged[[c + "_pred" for c in cols]].values

print("MAE:", round(mean_absolute_error(y_true, y_pred), 4))
print("MSE:", round(mean_squared_error(y_true, y_pred), 4))
print("R2:", round(r2_score(y_true, y_pred), 4))
print("Saved outputs/team_stats_diff_percent.csv")
