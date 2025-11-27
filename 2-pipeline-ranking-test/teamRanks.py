import math
import random

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVR


class TeamRanks:
    def __init__(self, training_df):
        self.encoders = {}
        training_df = self.preprocessTraining(training_df)
        model = TeamRanks.trainModel(training_df)
        self.model = model

    def preprocessTraining(self, df):
        df = df.copy()
        self.encoders = {}
        # Convert string data to a number
        for column in ["tmID", "confID"]:
            self.encoders[column] = LabelEncoder()
            df.loc[:, column] = self.encoders[column].fit_transform(
                df[column].astype(str)
            )

        return df

    def preprocessInput(self, df):
        df = df.copy()
        for column in ["tmID", "confID"]:
            df.loc[:, column] = self.encoders[column].fit_transform(
                df[column].astype(str)
            )

        return df

    @staticmethod
    def filterFeatures(df):
        return df[
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

    @staticmethod
    def filterTargets(df):
        return df["rank"]

    @staticmethod
    def trainModel(training_df):
        features = TeamRanks.filterFeatures(training_df)
        target = TeamRanks.filterTargets(training_df)
        model = NuSVR()
        model.fit(features, target)
        return model

    @staticmethod
    def __convertRegressionToClassification(test_df, results):
        # Get ranks from the float `rank`
        filtered_df = test_df[["year", "tmID", "confID"]].copy()
        filtered_df["rank_score"] = results

        years = []
        tmIDs = []
        confIDs = []
        rank_scores = []

        years = []
        tmIDs = []
        confIDs = []
        rank_scores = []

        for year in sorted(filtered_df["year"].unique()):
            for confID in sorted(filtered_df["confID"].unique()):
                year_results = filtered_df[
                    (filtered_df["year"] == year) & (filtered_df["confID"] == confID)
                ]
                sorted_year_results = year_results.sort_values("rank_score")
                for rank, tmID in enumerate(sorted_year_results["tmID"]):
                    years.append(year)
                    confIDs.append(confID)
                    tmIDs.append(tmID)
                    rank_scores.append(rank + 1)

        classification_ranks = pd.DataFrame(
            {
                "year": years,
                "tmID": tmIDs,
                "confID": confIDs,
                "rank": rank_scores,
            }
        )

        # Convert regression prediction into a classification prediction
        target_prediction = []
        year_tm_test = test_df[["year", "tmID", "confID"]]
        for row in year_tm_test.iterrows():
            year = row[1]["year"]
            tmID = row[1]["tmID"]
            confID = row[1]["confID"]
            rank = classification_ranks.loc[
                (classification_ranks["year"] == year)
                & (classification_ranks["tmID"] == tmID)
                & (classification_ranks["confID"] == confID),
                "rank",
            ].values[0]
            target_prediction.append(rank)

        return (classification_ranks, target_prediction)

    def generateResults(self, input_df):
        filtered_df = TeamRanks.filterFeatures(input_df)
        prediction = self.model.predict(filtered_df)
        (classification_ranks, _) = TeamRanks.__convertRegressionToClassification(
            input_df, prediction
        )
        classification_ranks["confID"] = self.encoders["confID"].inverse_transform(
            classification_ranks["confID"].astype(int)
        )
        classification_ranks["tmID"] = self.encoders["tmID"].inverse_transform(
            classification_ranks["tmID"].astype(int)
        )
        return classification_ranks

    def testModel(self, test_df):
        input_df = TeamRanks.filterFeatures(test_df)
        results = TeamRanks.filterTargets(test_df)
        prediction = self.model.predict(input_df)
        (_, prediction) = TeamRanks.__convertRegressionToClassification(
            test_df, prediction
        )
        print("Classification report: \n", classification_report(results, prediction))


def main():
    df = pd.read_csv("../database/final/teams.csv")

    # Divide dataframe into years and select some for testing
    years = df["year"].unique().tolist()
    years = random.sample(years, len(years))
    test_size = 0.2
    ty_len = math.ceil(len(years) * test_size)
    test_years = years[:ty_len]
    test_mask = df["year"].isin(test_years)

    training_df = df[~test_mask]
    model = TeamRanks(training_df)

    test_df = df[test_mask]
    model.testModel(test_df)


if __name__ == "__main__":
    main()
