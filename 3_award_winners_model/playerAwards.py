import math
import random

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import NuSVR


class PlayerAwards:
    def __init__(self, players_df, awards_df, player_encoder):
        players_df = players_df.copy()
        awards_df = awards_df.copy()
        self.encoders = {
            "playerID": player_encoder,
            "award": LabelEncoder().fit(PlayerAwards.validAwards()),
        }
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
        
    def preprocessTest(self, players_df):
        players_df["playerID"] = self.encoders["playerID"].transform(players_df["playerID"])
        return players_df


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
        model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            subsample=0.8
        ))
        model.fit(features, target)
        return model

    def convertRegressionToClassification(self, test_df, prediction_array):
        awards = [f"award_{self.encoders["award"].transform([name])[0]}" for name in PlayerAwards.validAwards()]
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
                player_col.append(int(winner_player))
        
        result =  pd.DataFrame({
            "playerID": player_col,
            "award": award_col,
            "year": year_col,
        })
                
        return result

    def generateResults(self, input_df):
        filtered_df = PlayerAwards.filterFeatures(input_df)
        prediction = self.model.predict(filtered_df)
        classification_df = self.convertRegressionToClassification(input_df, prediction)
        for column in ["playerID", "award"]:
            classification_df[column] = self.encoders[column].inverse_transform(classification_df[column])
        return classification_df

    def testModel(self, test_df, true_df):
        input_df = PlayerAwards.filterFeatures(test_df)
        prediction = self.model.predict(input_df)
        classification_df = self.convertRegressionToClassification(test_df, prediction)
        
        # Remove awards not given in the true_df
        player_answer = []
        player_prediction = []
        for idx, row in classification_df.iterrows():
            award = row["award"]
            year = row["year"]
            prediction = row["playerID"]
            answer = true_df[(true_df["award"] == award) & (true_df["year"] == year)]["playerID"].values
            if len(answer) == 1:
                player_answer.append(answer[0])
                player_prediction.append(prediction)
        
        print(player_prediction)
        print(player_answer)
        print("Classification Report: \n", classification_report(player_answer, player_prediction))


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
    player_encoder = LabelEncoder().fit(players_df["playerID"])
    model = PlayerAwards(training_players_df, training_awards_df, player_encoder)

    test_players_df = players_df[player_test_mask].reset_index(drop=True)
    test_awards_df = awards_df[awards_test_mask].reset_index(drop=True)
    test_df = model.preprocessTraining(test_players_df, test_awards_df)
    model.testModel(test_df, test_awards_df)


if __name__ == "__main__":
    main()
