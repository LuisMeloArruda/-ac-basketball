import math
import random

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVR

# 1. Load data
df = pd.read_csv("../database/final/teams.csv")

# 2. Preprocessing

# Convert string data to a number
for column in df:
    if df.dtypes[column] == object:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column].astype(str))

# Remove columns
df.drop(
    [
        "franchID",
        "playoff",
        "firstRound",
        "semis",
        "finals",
        "name",
        "won",
        "lost",
        "GP",
        "homeW",
        "homeL",
        "awayW",
        "awayL",
        "confW",
        "confL",
        "attend",
        "arena",
    ],
    axis=1,
    inplace=True,
)

# 3. Model
model = NuSVR()

# 4. Separate training and test data

# 4.1. Select years for testing
years = df["year"].unique().tolist()
years = random.sample(years, len(years))
test_size = 0.2
ty_len = math.ceil(len(years) * test_size)
test_years = years[:ty_len]

# 4.2. Divide train and test data
mask = df["year"].isin(test_years)
features = df[
    [
        "o_fgm",
        "o_fga",
        "o_ftm",
        "o_fta",
        "o_3pm",
        "o_oreb",
        "o_dreb",
        "o_reb",
        "o_asts",
        "o_pf",
        "o_stl",
        "o_to",
        "o_blk",
        "o_pts",
        "d_fga",
        "d_ftm",
        "d_3pm",
        "d_3pa",
        "d_oreb",
        "d_dreb",
        "d_asts",
        "d_pf",
        "d_stl",
        "d_to",
        "d_blk",
        "d_pts",
    ]
]
target = df["rank"]

features_train = features[~mask]
features_test = features[mask]
target_train = target[~mask]
target_test = target[mask]

# 5. Train model
model.fit(features_train, target_train)

# 6. Test model
target_prediction = model.predict(features_test)

# 7. Get ranks from the float `rank`
results = df[["year", "tmID", "confID"]]
results = results[mask]
results["rank_score"] = target_prediction

years = []
tmIDs = []
confIDs = []
rank_scores = []

for year in sorted(results["year"].unique()):
    for confID in sorted(results["confID"].unique()):
        year_results = results[(results["year"] == year) & (results["confID"] == confID)]
        sorted_year_results = year_results.sort_values("rank_score")
        for (rank, tmID) in enumerate(sorted_year_results["tmID"]):
            years.append(year)
            confIDs.append(confID)
            tmIDs.append(tmID)
            rank_scores.append(rank + 1)
        
classification_ranks = pd.DataFrame({
    "year": years,
    "tmID": tmIDs,
    "confID": confIDs,
    "rank": rank_scores,
})

# Convert regression prediction into a classification prediction
target_prediction = []
year_tm_test = df[["year", "tmID", "confID"]][mask]
for row in year_tm_test.iterrows():
    year = row[1]["year"]
    tmID = row[1]["tmID"]
    confID = row[1]["confID"]
    rank = classification_ranks.loc[(classification_ranks["year"] == year) & (classification_ranks["tmID"] == tmID) & (classification_ranks["confID"] == confID), "rank"].values[0]
    target_prediction.append(rank)
    

print("Clasification report: \n", classification_report(target_test, target_prediction))
