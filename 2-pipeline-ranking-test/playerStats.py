import math
import random

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder


class PlayerStats:
    def __init__(self, training_df):
        self.encoders = {}
        self.known_players = []
        training_df = self.preprocessTraining(training_df)
        model = PlayerStats.generateModel(training_df)
        self.model = model

    def preprocessTraining(self, df):
        # Convert string data to a number
        for column in ["playerID", "tmID"]:
            self.encoders[column] = LabelEncoder()
            df.loc[:, column] = self.encoders[column].fit_transform(df[column].astype(str))

        # Create Sparse Multi-Hot teammate matrix
        self.known_players = df["playerID"].unique()
        player_count = len(self.known_players)
        rows_count = df.shape[0]

        teammate_matrix = lil_matrix((rows_count, player_count), dtype=np.int8)

        for idx, row in df.iterrows():
            tmID, year, player = row["tmID"], row["year"], row["playerID"]

            teammates = df[
                (df["tmID"] == tmID) & (df["year"] == year) & (df["playerID"] != player)
            ]["playerID"]

            for teammate in teammates:
                teammate_matrix[idx, teammate] = 1

        # Convert sparse matrix to DataFrame
        teammate_df = pd.DataFrame(
            teammate_matrix.toarray(), columns=[f"teammate_{pid}" for pid in self.known_players]
        )

        # Merge with original DF
        df = pd.concat([df, teammate_df], axis=1)

        # Remove useless columns
        df.drop("stint", axis=1, inplace=True)

        return df
    
    def preprocessInput(self, df):
        # Convert string data to a number
        for column in ["playerID", "tmID"]:
            df.loc[:, column] = self.encoders[column].fit_transform(df[column].astype(str))

        # Create Sparse Multi-Hot teammate matrix
        player_count = len(self.known_players)
        rows_count = df.shape[0]

        teammate_matrix = lil_matrix((rows_count, player_count), dtype=np.int8)

        for idx, row in df.iterrows():
            tmID, year, player = row["tmID"], row["year"], row["playerID"]

            teammates = df[
                (df["tmID"] == tmID) & (df["year"] == year) & (df["playerID"] != player) & (df["playerID"].isin(self.known_players))
            ]["playerID"]

            for teammate in teammates:
                teammate_matrix[idx, teammate] = 1

        # Convert sparse matrix to DataFrame
        teammate_df = pd.DataFrame(
            teammate_matrix.toarray(), columns=[f"teammate_{pid}" for pid in self.known_players]
        )

        # Merge with original DF
        df = pd.concat([df, teammate_df], axis=1)

        # Remove useless columns
        df.drop("stint", axis=1, inplace=True)

        return df

    @staticmethod
    def filterFeatures(df):
        teammate_columns = [
            df_col for df_col in df.columns if df_col.startswith("teammate_")
        ]
        return df[["playerID", "tmID"] + teammate_columns]

    @staticmethod
    def filterTargets(df):
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
    def generateModel(training_df):
        features = PlayerStats.filterFeatures(training_df)
        target = PlayerStats.filterTargets(training_df)
        model = MultiOutputRegressor(RandomForestRegressor(), n_jobs=1)
        model.fit(features, target)
        return model

    def generateResult(self, input_df):
        filtered_df = PlayerStats.filterFeatures(input_df)
        target_prediction = self.model.predict(filtered_df)

        # Build prediction DataFrame
        pred_df = pd.DataFrame(
            target_prediction,
            columns=[
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
            ],
        )

        # Restore real Player IDs
        pred_df["playerID"] = self.encoders["playerID"].inverse_transform(
            input_df["playerID"].astype(int)
        )

        # Restore years and team IDs
        pred_df["year"] = input_df["year"]
        pred_df["tmID"] = self.encoders["tmID"].inverse_transform(
            input_df["tmID"].astype(int)
        )

        # reorder columns: playerID, year, tmID, stats
        pred_df = pred_df[
            ["playerID", "year", "tmID"]
            + [c for c in pred_df.columns if c not in ["playerID", "year", "tmID"]]
        ]

        pred_df.to_csv("outputs/predicted_players_stats.csv", index=False)
        print("\nSaved predicted_players_stats.csv")

        return pred_df

    def testModel(self, input_df, results):
        prediction = self.model.predict(input_df)
        print("Mean Absolute Error: ", mean_absolute_error(results, prediction))
        print("Mean Squared Error: ", mean_squared_error(results, prediction))
        print("R Squared Score: ", r2_score(results, prediction))


def main():
    df = pd.read_csv("../database/final/players_teams.csv")

    # Divide dataframe into years and select some for testing
    years = df["year"].unique().tolist()
    random.shuffle(years)
    test_size = 0.1
    test_years = years[: math.ceil(len(years) * test_size)]
    test_mask = df["year"].isin(test_years)
    
    training_df = df[~test_mask]
    training_df.reset_index(drop=True, inplace=True)
    model = PlayerStats(training_df)

    test_df = df[test_mask]
    test_df.reset_index(drop=True, inplace=True)
    test_df = model.preprocessInput(test_df)
    features = PlayerStats.filterFeatures(test_df)
    targets = PlayerStats.filterTargets(test_df)
    model.testModel(features, targets)


if __name__ == "__main__":
    main()
