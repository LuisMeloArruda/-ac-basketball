import math
import random
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# 1. Load data
df = pd.read_csv("../database/final/players_teams.csv")

# 2. Preprocessing

# Convert string data to a number
for column in df:
    if df.dtypes[column] == object:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column].astype(str))

# Create `teamates` column as an array of playerIDs from the same team
# teamID -> Array(PlayerID)
team_players_dict = {}

for _, row in df.iterrows():
    teamID = row["tmID"]
    year = row["year"]
    playerID = row["playerID"]

    if (teamID, year) not in team_players_dict:
        team_players_dict[(teamID, year)] = []
    team_players_dict[(teamID, year)].append(playerID)

teammates = []
for _, row in df.iterrows():
    teamID = row["tmID"]
    year = row["year"]
    teammates.append(team_players_dict[(teamID, year)])

df['teammates'] = teammates

# Remove column `stint`
df.drop('stint', axis=1, inplace=True)


# TODO: Convert `teammates` to a set of columns with playerIDs and 0/1 values


# 3. Pipeline
pipeline = Pipeline(steps=[
    ('model', RandomForestClassifier(random_state=42))
])

# 4. Separate traning and test data

# 4.1. Select years for testing
years = df['year'].unique().tolist()
years = random.sample(years, len(years))
test_size = 0.2
ty_len = math.ceil(len(years) * test_size)
test_years = years[:ty_len]

# 4.2. Divide train and test data 
features_train = []
features_test = []
target_train = []
target_test = []

for _, row in df.iterrows():
    feature_line = row[["playerID", "teammates"]]
    target_line = df.drop(columns=["playerID", "teammates"])
    if row["year"] in test_years:
        target_test.append(target_line)
        features_test.append(feature_line)
    else:
        target_train.append(target_line)
        features_train.append(feature_line)

# 5. Train model
pipeline.fit(features_train, target_train)

# 6. Test model
target_prediction = pipeline.predict(features_test)
print(classification_report(target_test, target_prediction))