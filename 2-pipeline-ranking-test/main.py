import math
import random
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load data

# TODO: Desnormalize data of all .csv files into one DataFrame
df = pd.read_csv("../database/cleaned/teams.csv")

# 2. Preprocessing
for column in df:
    if df.dtypes[column] == object:
        # TODO: is necessary to hold the encoder for each column?
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])

# TODO: Improve preprocessing

# 3. Pipeline

# TODO: Explore different models
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

for line in df.iterrows():
    line = line[1]
    target_line = line["rank"]
    feature_line = line.drop(columns="rank")
    if line["year"] in test_years:
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
