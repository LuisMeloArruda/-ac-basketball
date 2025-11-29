import math
import random

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVR


class PlayerAwards:
    def __init__(self, players_df, awards_df):
        players_df = players_df.copy()
        awards_df = awards_df.copy()
        self.encoders = {}
        training_df = self.preprocess(players_df, awards_df)
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

    def preprocess(self, players_df, awards_df):
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
            
            try:
                player_year_idx = players_df[(players_df["playerID"] == player) & (players_df["year"] == year)].index[0]
            except IndexError:
                continue
            
            awards_matrix[player_year_idx, award] = 1
        
        matrix_df = pd.DataFrame(
            awards_matrix.toarray(),
            columns=[f"award_{id}" for id in awards],
        )

        df = pd.concat([players_df, matrix_df], axis=1)
        
        return df

    @staticmethod
    def filterFeatures(df):
        return df[
            [
                "fgAttempted",
                "fgMade",
                "ftMade",
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
                "threeMade",
                "dq",
            ]
        ]

    @staticmethod
    def filterTargets(df):
        return df[[col for col in df.columns if col.startswith("award_")]]

    @staticmethod
    def trainModel(training_df):
        features = PlayerAwards.filterFeatures(training_df)
        target = PlayerAwards.filterTargets(training_df)
        model = MultiOutputRegressor(NuSVR())
        model.fit(features, target)
        return model

    @staticmethod
    def __convertRegressionToClassification(test_df, prediction_array):
        awards = [col for col in test_df.columns if col.startswith("award_")]
        prediction_df = pd.DataFrame(prediction_array, columns=awards)
        df = pd.concat([test_df[["playerID", "year"]], prediction_df], axis=1)
        years = df["year"].unique()
        
        year_col = []
        award_col = []
        player_col = []
        
        for year in years:
            for award in awards:
                year_df = df[df["year"] == year]
                award_year_df = year_df[["playerID", award]]
                sorted_award_year_df = award_year_df.sort_values(award, ascending=False)
                
                winner_player = sorted_award_year_df.iloc[0]["playerID"]
                
                year_col.append(year)
                award_col.append(int(award.replace("award_", "")))
                player_col.append(winner_player)
        
        return pd.DataFrame({
            "playerID": player_col,
            "award": award_col,
            "year": year_col,
        })

    def generateResults(self, input_df):
        filtered_df = PlayerAwards.filterFeatures(input_df)
        prediction = self.model.predict(filtered_df)
        classification_df = self.__convertRegressionToClassification(input_df, prediction)
        for column in ["playerID", "award"]:
            classification_df[column] = self.encoders[column].inverse_transform(classification_df[column])
        return classification_df

    def testModel(self, test_df, true_df):
        input_df = PlayerAwards.filterFeatures(test_df)
        prediction = self.model.predict(input_df)
        classification_df = self.__convertRegressionToClassification(test_df, prediction)
        
        # Remove awards not given in the true_df
        valid_awards_year = set(zip(true_df["award"], true_df["year"]))
        for idx, row in classification_df.iterrows():
            award_year = (row["award"], row["year"])
            print(award_year)
            if award_year not in valid_awards_year:
                classification_df = classification_df.drop(idx)
        classification_df.reset_index(drop=True)
        
        player_prediction = classification_df["playerID"]
        player_results = true_df["playerID"]
        print("Classification Report: \n", classification_report(player_results, player_prediction))


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

    test_players_df = players_df[player_test_mask].reset_index(drop=True)
    test_awards_df = awards_df[awards_test_mask].reset_index(drop=True)
    test_df = model.preprocess(test_players_df, test_awards_df)
    model.testModel(test_df, test_awards_df)


if __name__ == "__main__":
    main()
