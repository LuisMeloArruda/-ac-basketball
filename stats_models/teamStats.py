import math
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder


class TeamStats:
    def __init__(self, training_df):
        self.encoders = {}
        training_df = self.preprocessTraining(training_df)
        model = TeamStats.trainModel(training_df)
        self.model = model

    def preprocessTraining(self, training_df):
        df = training_df.copy()
        le = LabelEncoder()
        df["tmID"] = le.fit_transform(df["tmID"].astype(str))
        self.encoders = {"tmID": le}
        return df

    def preprocessInput(self, input_df):
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

        team["tmID"] = self.encoders["tmID"].transform(team["tmID"].astype(str))

        return team

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
                "o_pts",
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

    def generateResults(self, input_df):
        d_pred = self.model.predict(input_df)
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
        df_output["tmID"] = self.encoders["tmID"].inverse_transform(df_output["tmID"])

        return df_output

    def testModel(self, input_df, result_df):
        pred = self.generateResults(input_df)

        # merge predicted + real
        merged = pred.merge(result_df, on=["tmID", "year"], suffixes=("_pred", "_real"))

        # offensive stats
        off_cols = [
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

        # defensive stats (auto-detected)
        def_cols = [c for c in result_df.columns if c.startswith("d_")]

        # full list
        cols = off_cols + def_cols

        diff = pd.DataFrame()
        diff["tmID"] = merged["tmID"]
        diff["year"] = merged["year"]

        # compute % error
        for c in cols:
            real_vals = merged[c + "_real"]
            pred_vals = merged[c + "_pred"]

            abs_err = (real_vals - pred_vals).abs()

            percent_err = np.where(real_vals != 0, (abs_err / real_vals.abs()) * 100, 0)

            diff[c + "_err_percent"] = percent_err.round(2)

        # model-wide metrics
        y_true = merged[[c + "_real" for c in cols]].values
        y_pred = merged[[c + "_pred" for c in cols]].values

        print("MAE:", round(mean_absolute_error(y_true, y_pred), 4))
        print("MSE:", round(mean_squared_error(y_true, y_pred), 4))
        print("R2:", round(r2_score(y_true, y_pred), 4))


def main():
    df = pd.read_csv("../database/final/teams.csv")
    input_df = pd.read_csv("../database/final/players_teams.csv")

    # Divide dataframe into years and select some for testing
    years = df["year"].unique().tolist()
    years = random.sample(years, len(years))
    test_size = 0.1
    ty_len = math.ceil(len(years) * test_size)
    test_years = years[:ty_len]

    training_test_mask = df["year"].isin(test_years)
    training_df = df[~training_test_mask]
    model = TeamStats(training_df)

    input_df = model.preprocessInput(input_df)
    input_test_mask = input_df["year"].isin(test_years)
    input_df = input_df[input_test_mask]

    input_df = TeamStats.filterFeatures(input_df)
    result_df = df[training_test_mask]

    model.testModel(input_df, result_df)


if __name__ == "__main__":
    main()
