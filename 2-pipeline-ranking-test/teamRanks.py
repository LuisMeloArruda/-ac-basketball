import math
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

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
model = RandomForestClassifier()

# 4. Separate training and test data

# 4.1. Select years for testing
years = df["year"].unique().tolist()
years = random.sample(years, len(years))
test_size = 0.2
ty_len = math.ceil(len(years) * test_size)
# test_years = years[:ty_len]
test_years = [7]

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
        "d_fgm",
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
print(confusion_matrix(target_test, target_prediction))
print(classification_report(target_test, target_prediction))
