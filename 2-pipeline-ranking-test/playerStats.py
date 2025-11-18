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
for column in df:
    if df.dtypes[column] == object:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column].astype(str))

# Sparse Multi-Hot matrix to encode teammates
player_count = len(df["playerID"].unique())
rows_count = df["playerID"].shape[0]
teammate_matrix = lil_matrix((rows_count, player_count), dtype=np.int8)

for idx, row in df.iterrows():
    tmID, year, player = row["tmID"], row["year"], row["playerID"]
    teammates = df[
        (df["tmID"] == tmID) & (df["year"] == year) & (df["playerID"] != player)
    ]["playerID"]
    for teammate in teammates:
        teammate_matrix[idx, teammate] = 1

teammate_df = pd.DataFrame(
    teammate_matrix.toarray(),
    # index=['playerID'],
    columns=[f"teammate_{id}" for id in df["playerID"].unique()],
)

# Merge datframes
df = pd.concat([df, teammate_df], axis=1)

# Remove columns
df.drop("stint", axis=1, inplace=True)
df.drop("tmID", axis=1, inplace=True)

# 3. Model
model = MultiOutputRegressor(RandomForestRegressor(), n_jobs=1)

# 4. Separate training and test data

# 4.1. Select years for testing
years = df["year"].unique().tolist()
years = random.sample(years, len(years))
test_size = 0.2
ty_len = math.ceil(len(years) * test_size)
test_years = years[:ty_len]

# 4.2. Divide train and test data
mask = df["year"].isin(test_years)
features = df[["playerID"] + [f"teammate_{id}" for id in df["playerID"].unique()]]
target = df[
    [
        "GP",
        "GS",
        "minutes",
        "points",
        "oRebounds",
        "dRebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "PF",
        "ftAttempted",
        "threeAttempted",
        "dq",
    ]
]

features_train = features[~mask]
features_test = features[mask]
target_train = target[~mask]
target_test = target[mask]

# 5. Train model
model.fit(features_train, target_train)

# 6. Test model
target_prediction = model.predict(features_test)
print("Mean Absolute Error: ", mean_absolute_error(target_test, target_prediction))
print("Mean Squared Error: ", mean_squared_error(target_test, target_prediction))
print("R Squared Score: ", r2_score(target_test, target_prediction))
