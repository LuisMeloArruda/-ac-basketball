import math
import random

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVR


class PlayerAwards:
    def __init__(self, players_df, awards_df):
        players_df = players_df.copy()
        awards_df = awards_df.copy()
        self.encoders = {}
        training_df = self.preprocessTraining(players_df, awards_df)
        model = PlayerAwards.trainModel(training_df)
        self.model = model

    @staticmethod
    def validAwards():
        return [
            "All-Star Game Most Valuable Player",
            "Defensive Player of the Year",
            "Kim Perrot Sportsmanship Award",
            "Most Improved Player",
            "Most Valuable Player",
            "Rookie of the Year",
            "Sixth Woman of the Year",
            "WNBA Final Most Valuable Player",
        ]

    def preprocessTraining(self, players_df, awards_df):
        # Convert strings to categorical numbers
        self.encoders["award"] = LabelEncoder().fit(PlayerAwards.validAwards())
        self.encoders["playerID"] = LabelEncoder().fit(players_df["playerID"])
        awards = self.encoders["award"].transform(PlayerAwards.validAwards())
        awards_df["award"] = self.encoders["award"].transform(awards_df["award"])
        awards_df["playerID"] = self.encoders["playerID"].transform(awards_df["playerID"])
        players_df["playerID"] = self.encoders["playerID"].transform(players_df["playerID"])
        
        # Denormalize datasets (merge into one)
        row_count = len(players_df)
        award_count = len(PlayerAwards.validAwards())
        awards_matrix = lil_matrix((row_count, award_count), dtype=np.int8)
        for (_, row) in awards_df.iterrows():
            player = row["playerID"]
            year = row["year"]
            award = row["award"]
            # Find the index for the player-year combination
            try:
                player_year_idx = players_df[(players_df["playerID"] == player) & (players_df["year"] == year)].index[0]
            except IndexError:
                continue  # skip if player-year not found
            awards_matrix[player_year_idx, award] = 1
        
        matrix_df = pd.DataFrame(
            awards_matrix.toarray(),
            columns=[f"award_{id}" for id in awards],
        )

        # Merge with original DF
        df = pd.concat([players_df, matrix_df], axis=1)
        
        # DEBUG: Fix bug of incorrect assignment of awards to the wrong players
        df.to_csv("test.csv", index=False)

    def preprocessInput(self, df):
        todo()

    @staticmethod
    def filterFeatures(df):
        return df[
            [
                "todo",
            ]
        ]

    @staticmethod
    def filterTargets(df):
        return df[["todo"]]

    @staticmethod
    def trainModel(training_df):
        features = PlayerAwards.filterFeatures(training_df)
        target = PlayerAwards.filterTargets(training_df)
        model = todo()
        model.fit(features, target)
        return model


    def generateResults(self, input_df):
        filtered_df = PlayerAwards.filterFeatures(input_df)
        prediction = self.model.predict(filtered_df)
        return prediction

    def testModel(self, test_df):
        input_df = PlayerAwards.filterFeatures(test_df)
        results = PlayerAwards.filterTargets(test_df)
        prediction = self.model.predict(input_df)
        print("Classification report: \n", classification_report(results, prediction))


def main():
    players_df = pd.read_csv("../database/final/players_teams.csv")
    awards_df = pd.read_csv("../database/final/awards_players.csv")
    awards_df = awards_df[awards_df["award"].isin(PlayerAwards.validAwards())]

    # Divide dataframe into years and select some for testing
    years = players_df["year"].unique().tolist()
    years = random.sample(years, len(years))
    test_size = 0.2
    ty_len = math.ceil(len(years) * test_size)
    test_years = years[:ty_len]
    player_test_mask = players_df["year"].isin(test_years)
    awards_test_mask = awards_df["year"].isin(test_years)

    training_players_df = players_df[~player_test_mask].reset_index(drop=True)
    training_awards_df = awards_df[~awards_test_mask].reset_index(drop=True)
    model = PlayerAwards(training_players_df, training_awards_df)

    test_df = df[test_mask]
    model.testModel(test_df)


if __name__ == "__main__":
    main()
