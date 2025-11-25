import math
import random
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

# 1. Load data
df = pd.read_csv("../database/final/players_teams.csv")

# 2. Preprocessing

# Convert string data to a number
encoders = {}
for column in df:
    if df.dtypes[column] == object:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column].astype(str))
        encoders[column] = label_encoder

# 2.1 Create Sparse Multi-Hot teammate matrix
player_ids = df["playerID"].unique()
player_count = len(player_ids)
rows_count = df.shape[0]

id_to_index = {pid: i for i, pid in enumerate(player_ids)}

teammate_matrix = lil_matrix((rows_count, player_count), dtype=np.int8)

for idx, row in df.iterrows():
    tmID, year, player = row["tmID"], row["year"], row["playerID"]

    teammates = df[
        (df["tmID"] == tmID) &
        (df["year"] == year) &
        (df["playerID"] != player)
    ]["playerID"]

    for teammate in teammates:
        teammate_matrix[idx, id_to_index[teammate]] = 1

# Convert sparse matrix to DataFrame
teammate_df = pd.DataFrame(
    teammate_matrix.toarray(),
    columns=[f"teammate_{pid}" for pid in player_ids]
)

# Merge with original DF
df = pd.concat([df, teammate_df], axis=1)

# Remove useless columns
df.drop("stint", axis=1, inplace=True)


# 3. Model
model = MultiOutputRegressor(RandomForestRegressor(), n_jobs=1)

# 4. Split train/test by YEARS

years = df["year"].unique().tolist()
random.shuffle(years)

test_size = 0.1
test_years = years[:math.ceil(len(years) * test_size)]

mask = df["year"].isin(test_years)

features = df[
    ["playerID", "tmID"] +
    [f"teammate_{pid}" for pid in player_ids]
]

target = df[
    [
        "fgAttempted", "fgMade", "ftMade",
        "GP", "GS", "minutes", "points",
        "oRebounds", "dRebounds",
        "assists", "steals", "blocks",
        "turnovers", "PF",
        "ftAttempted", "threeMade", "dq"
    ]
]

features_train = features[~mask]
features_test = features[mask]
target_train = target[~mask]
target_test = target[mask]

# 5. Train model
model.fit(features_train, target_train)

# 6. Evaluate
target_prediction = model.predict(features_test)

print("Mean Absolute Error: ", mean_absolute_error(target_test, target_prediction))
print("Mean Squared Error: ", mean_squared_error(target_test, target_prediction))
print("R Squared Score: ", r2_score(target_test, target_prediction))

# 7. Build prediction DataFrame
pred_df = pd.DataFrame(
    target_prediction,
    columns=[
        "fgAttempted", "fgMade", "ftMade",
        "GP", "GS", "minutes", "points",
        "oRebounds", "dRebounds",
        "assists", "steals", "blocks",
        "turnovers", "PF",
        "ftAttempted", "threeMade", "dq"
    ]
)

# Restore real Player IDs
pred_df["playerID"] = encoders["playerID"].inverse_transform(
    features_test["playerID"].values
)

# Restore years and team IDs
pred_df["year"] = df.loc[mask, "year"].values
pred_df["tmID"] = encoders["tmID"].inverse_transform(
    df.loc[mask, "tmID"].values
)

# reorder columns: playerID, year, tmID, stats
pred_df = pred_df[
    ["playerID", "year", "tmID"] +
    [c for c in pred_df.columns if c not in ["playerID", "year", "tmID"]]
]

pred_df.to_csv("outputs/predicted_players_stats.csv", index=False)
print("\nSaved predicted_players_stats.csv")
