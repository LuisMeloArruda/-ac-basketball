import math
import random

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder


class TeamStats:
    @staticmethod
    def preprocessTraining(training_df):
        le = LabelEncoder()
        training_df["tmID"] = le.fit_transform(training_df["tmID"].astype(str))
        return (training_df, le)

    @staticmethod
    def preprocessInput(input_df, encoder):
        team = (
            input_df.groupby(["tmID", "year"])
            .agg(
                {
                    "fgAttempted": "sum",
                    "fgMade": "sum",
                    "ftMade": "sum",
                    "points": "sum",
                    "oRebounds": "sum",
                    "dRebounds": "sum",
                    "assists": "sum",
                    "steals": "sum",
                    "blocks": "sum",
                    "turnovers": "sum",
                    "PF": "sum",
                    "ftAttempted": "sum",
                    "threeMade": "sum",
                    "dq": "sum",
                }
            )
            .reset_index()
        )

        team["o_fgm"] = team["fgMade"]
        team["o_fga"] = team["fgAttempted"]
        team["o_ftm"] = team["ftMade"]
        team["o_fta"] = team["ftAttempted"]
        team["o_3pm"] = team["threeMade"]
        team["o_oreb"] = team["oRebounds"]
        team["o_dreb"] = team["dRebounds"]
        team["o_reb"] = team["oRebounds"] + team["dRebounds"]
        team["o_asts"] = team["assists"]
        team["o_pf"] = team["PF"]
        team["o_stl"] = team["steals"]
        team["o_to"] = team["turnovers"]
        team["o_blk"] = team["blocks"]
        team["o_pts"] = team["points"]

        cols = [
            "tmID",
            "year",
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
        ]

        team = team[cols]

        team = team.round(2)

        team["tmID"] = encoder.transform(team["tmID"].astype(str))

        return (team, encoder)

    @staticmethod
    def filterFeatures(df):
        return df[
            [
                "tmID",
                "year",
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
            ]
        ]

    @staticmethod
    def filterTargets(df):
        return df[
            [
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
    def trainModel(training_df):
        X = TeamStats.filterFeatures(training_df)
        Y = TeamStats.filterTargets(training_df)
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        )
        model.fit(X, Y)
        return model

    @staticmethod
    def generateResults(model, input_df, encoder):
        d_pred = model.predict(input_df)
        df_def = pd.DataFrame(
            d_pred,
            columns=[
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
            ],
        )
        df_def = df_def.round(2)

        df_output = pd.concat([input_df.reset_index(drop=True), df_def], axis=1)

        df_output.to_csv("./outputs/predicted_team_stats.csv", index=False)

        print("saved ./outputs/predicted_team_stats.csv")

    @staticmethod
    def testModel(model, input_df, result_df):
        print(input_df.head())
        print(result_df.head())
        prediction = model.predict(input_df)
        print(prediction.head())
        print("Mean Absolute Error: ", mean_absolute_error(result_df, prediction))
        print("Mean Squared Error: ", mean_squared_error(result_df, prediction))
        print("R Squared Score: ", r2_score(result_df, prediction))


def main():
    training_df = pd.read_csv("../database/final/teams.csv")
    input_df = pd.read_csv("../database/final/players_teams.csv")

    # Divide dataframe into years and select some for testing
    years = training_df["year"].unique().tolist()
    years = random.sample(years, len(years))
    test_size = 0.2
    ty_len = math.ceil(len(years) * test_size)
    test_years = years[:ty_len]
    training_test_mask = training_df["year"].isin(test_years)
    input_test_mask = input_df["year"].isin(test_years)

    (training_df, encoder) = TeamStats.preprocessTraining(training_df)
    (input_df, encoder) = TeamStats.preprocessInput(input_df, encoder)
    training_df = training_df[~training_test_mask]
    
    input_df = input_df[input_test_mask]
    input_df = TeamStats.filterFeatures(input_df)
    result_df = training_df[training_test_mask]
    result_df = TeamStats.filterTargets(result_df)

    model = TeamStats.trainModel(training_df)
    TeamStats.testModel(model, input_df, result_df)


if __name__ == "__main__":
    main()


# team = (
#     df.groupby(["tmID", "year"])
#     .agg(
#         {
#             "fgAttempted": "sum",
#             "fgMade": "sum",
#             "ftMade": "sum",
#             "points": "sum",
#             "oRebounds": "sum",
#             "dRebounds": "sum",
#             "assists": "sum",
#             "steals": "sum",
#             "blocks": "sum",
#             "turnovers": "sum",
#             "PF": "sum",
#             "ftAttempted": "sum",
#             "threeMade": "sum",
#             "dq": "sum",
#         }
#     )
#     .reset_index()
# )

# team["o_fgm"] = team["fgMade"]
# team["o_fga"] = team["fgAttempted"]
# team["o_ftm"] = team["ftMade"]
# team["o_fta"] = team["ftAttempted"]
# team["o_3pm"] = team["threeMade"]
# team["o_oreb"] = team["oRebounds"]
# team["o_dreb"] = team["dRebounds"]
# team["o_reb"] = team["oRebounds"] + team["dRebounds"]
# team["o_asts"] = team["assists"]
# team["o_pf"] = team["PF"]
# team["o_stl"] = team["steals"]
# team["o_to"] = team["turnovers"]
# team["o_blk"] = team["blocks"]
# team["o_pts"] = team["points"]

# cols = [
#     "tmID",
#     "year",
#     "o_fgm",
#     "o_fga",
#     "o_ftm",
#     "o_fta",
#     "o_3pm",
#     "o_oreb",
#     "o_dreb",
#     "o_reb",
#     "o_asts",
#     "o_pf",
#     "o_stl",
#     "o_to",
#     "o_blk",
#     "o_pts",
# ]

# team = team[cols]

# team = team.round(2)

# df_pred_players = pd.read_csv("./outputs/predicted_players_stats.csv")
# target_year = int(df_pred_players["year"].iloc[0])

# print("Target year:", target_year)


# df_hist = pd.read_csv("../database/final/teams.csv")
# df_hist = df_hist[df_hist["year"] != target_year]

# o_cols = [
#     "o_fgm",
#     "o_fga",
#     "o_ftm",
#     "o_fta",
#     "o_3pm",
#     "o_oreb",
#     "o_dreb",
#     "o_reb",
#     "o_asts",
#     "o_pf",
#     "o_stl",
#     "o_to",
#     "o_blk",
# ]

# d_cols = [
#     "d_fga",
#     "d_ftm",
#     "d_3pm",
#     "d_3pa",
#     "d_oreb",
#     "d_dreb",
#     "d_asts",
#     "d_pf",
#     "d_stl",
#     "d_to",
#     "d_blk",
#     "d_pts",
# ]

# X = df_hist[["tmID", "year"] + o_cols].copy()
# Y = df_hist[d_cols].copy()


# le = LabelEncoder()
# X["tmID"] = le.fit_transform(X["tmID"].astype(str))

# model = MultiOutputRegressor(
#     RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
# )

# model.fit(X, Y)


# team_encoded = team.copy()
# team_encoded["tmID"] = le.transform(team_encoded["tmID"].astype(str))

# X_future = team_encoded[["tmID", "year"] + o_cols]

# d_pred = model.predict(X_future)

# df_def = pd.DataFrame(d_pred, columns=d_cols)
# df_def = df_def.round(2)

# df_output = pd.concat([team.reset_index(drop=True), df_def], axis=1)

# df_output.to_csv("./outputs/predicted_team_stats.csv", index=False)

# print("saved ./outputs/predicted_team_stats.csv")
