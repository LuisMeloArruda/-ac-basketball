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
    @staticmethod
    def preprocess(df):
        # Convert string data to a number
        encoders = {}
        for column in df:
            if df.dtypes[column] == object:
                label_encoder = LabelEncoder()
                df.loc[:, column] = label_encoder.fit_transform(df[column].astype(str))
                encoders[column] = label_encoder

        # Create Sparse Multi-Hot teammate matrix
        player_ids = df["playerID"].unique()
        player_count = len(player_ids)
        rows_count = df.shape[0]

        id_to_index = {pid: i for i, pid in enumerate(player_ids)}

        teammate_matrix = lil_matrix((rows_count, player_count), dtype=np.int8)

        for idx, row in df.iterrows():
            tmID, year, player = row["tmID"], row["year"], row["playerID"]

            teammates = df[
                (df["tmID"] == tmID) & (df["year"] == year) & (df["playerID"] != player)
            ]["playerID"]

            for teammate in teammates:
                # print(rows_count, player_count, idx, id_to_index[teammate])
                teammate_matrix[idx, id_to_index[teammate]] = 1

        # Convert sparse matrix to DataFrame
        teammate_df = pd.DataFrame(
            teammate_matrix.toarray(), columns=[f"teammate_{pid}" for pid in player_ids]
        )

        # Merge with original DF
        df = pd.concat([df, teammate_df], axis=1)

        # Remove useless columns
        df.drop("stint", axis=1, inplace=True)

        return (df, encoders)

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

    @staticmethod
    def generateResult(model, input_df, encoders):
        target_prediction = model.predict(input_df)

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
        pred_df["playerID"] = encoders["playerID"].inverse_transform(
            input_df["playerID"]
        )

        # Restore years and team IDs
        pred_df["year"] = input_df["year"]
        pred_df["tmID"] = encoders["tmID"].inverse_transform(input_df["tmID"])

        # reorder columns: playerID, year, tmID, stats
        pred_df = pred_df[
            ["playerID", "year", "tmID"]
            + [c for c in pred_df.columns if c not in ["playerID", "year", "tmID"]]
        ]

        pred_df.to_csv("outputs/predicted_players_stats.csv", index=False)
        print("\nSaved predicted_players_stats.csv")

    @staticmethod
    def testModel(model, input_df, results):
        prediction = model.predict(input_df)
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

    (df, encoders) = PlayerStats.preprocess(df)
    training_df = df[~test_mask]
    model = PlayerStats.generateModel(training_df)

    test_df = df[test_mask]
    test_df.reset_index(drop=True, inplace=True)
    features = PlayerStats.filterFeatures(test_df)
    targets = PlayerStats.filterTargets(test_df)
    PlayerStats.testModel(model, features, targets)


if __name__ == "__main__":
    main()
