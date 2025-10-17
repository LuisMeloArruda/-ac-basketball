import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load data

# TODO: Desnormalize data of all .csv files into one DataFrame
df = pd.read_csv("../database/teams.csv")

# 2. Preprocessing
cols_to_drop = [
    "seeded",
    "divID",
    "tmORB", "tmDRB", "tmTRB",
    "opptmORB", "opptmDRB", "opptmTRB"
]
df = df.drop(columns=cols_to_drop)

for column in df:
    if df.dtypes[column] == object:
        # TODO: is necessary to hold the encoder for each column?
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])

# TODO: Improve preprocessing

# 3. Seperate target from data
target = df["rank"]
features = df.drop(columns="rank")

# 4. Pipeline

# TODO: Explore different models
pipeline = Pipeline(steps=[
    ('model', RandomForestClassifier(random_state=42))
])

# 5. Separate traning and test data

# TODO: Improve test divide
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 6. Train model
pipeline.fit(features_train, target_train)

# 7. Test model
target_prediction = pipeline.predict(features_test)
print(classification_report(target_test, target_prediction))
