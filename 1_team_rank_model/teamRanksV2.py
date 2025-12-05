import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class TeamRank:
    def __init__(self, teams_df):
        """
        Initialize the ranking model: copy the input dataframe,
        split teams by conference, and create a global encoder for team IDs.
        """

        self.teams_df = teams_df.copy()
        self.conferences = sorted(self.teams_df["confID"].unique())

        self.conf_groups = {
            conf: self.teams_df[self.teams_df["confID"] == conf].copy()
            for conf in self.conferences
        }

        self.encoder_tm = LabelEncoder()
        self.encoder_tm.fit(self.teams_df["tmID"].astype(str))


    def filter_valid_tmids(self, df, df_conf, year):
        """
        Keep only team IDs that belong to the given conference and year.
        Ensures that predictions align with the actual conference membership.
        """

        valid = df_conf[df_conf["year"] == year]["tmID"].astype(str).unique()
        df["tmID"] = df["tmID"].astype(str)
        return df[df["tmID"].isin(valid)].reset_index(drop=True)


    def buildFeaturesForYear(self, year, df_conf):
        """
        Load and combine OFFENSE, DEFENSE and WIN/LOSS prediction files
        for a specific year, applying conference/year filtering and encoding.
        Returns a dataframe of all features.
        """

        off_path = f"../stats_models/teams_offense_predictions/offense_predictions_year_{year}.csv"
        if not os.path.exists(off_path):
            return None

        off_df = pd.read_csv(off_path)
        off_df = self.filter_valid_tmids(off_df, df_conf, year)
        if off_df.empty:
            return None
        off_df["tmID_enc"] = self.encoder_tm.transform(off_df["tmID"])

        def_path = f"../stats_models/teams_defense_predictions/defense_predictions_year_{year}.csv"
        if not os.path.exists(def_path):
            return None

        def_df = pd.read_csv(def_path)
        def_df = self.filter_valid_tmids(def_df, df_conf, year)
        if def_df.empty:
            return None
        def_df["tmID_enc"] = self.encoder_tm.transform(def_df["tmID"])

        df = off_df.merge(def_df, on=["tmID", "tmID_enc", "year"], how="left")

        wl_path = f"../stats_models/teams_wl_predictions/wl_predictions_year_{year}.csv"
        if os.path.exists(wl_path):
            wl_df = pd.read_csv(wl_path)
            wl_df = self.filter_valid_tmids(wl_df, df_conf, year)
            if not wl_df.empty:
                wl_df["tmID_enc"] = self.encoder_tm.transform(wl_df["tmID"])
                df = df.merge(wl_df, on=["tmID", "tmID_enc", "year"], how="left")

        return df


    def buildTrainingSet(self, years, df_conf):
        """
        Build the training dataset: select previous years, encode team IDs,
        extract offensive, defensive, and win/loss features,
        and return X, y and the feature column list.
        """

        df = df_conf[df_conf["year"].isin(years)].copy()
        df["tmID_enc"] = self.encoder_tm.transform(df["tmID"].astype(str))

        feature_cols = [
            "o_fgm","o_fga","o_ftm","o_fta","o_3pm",
            "o_oreb","o_dreb","o_reb",
            "o_asts","o_pf","o_stl","o_to","o_blk","o_pts",
            "d_fgm","d_ftm","d_3pm","d_3pa",
            "d_oreb","d_dreb","d_asts",
            "d_pf","d_stl","d_to","d_blk","d_pts",
            "won", "lost",
        ]

        X = df[feature_cols].copy()
        y = df["rank"].astype(int)
        return X, y, feature_cols


    def trainModel(self, X, y):
        """
        Train a RandomForestRegressor to predict team rankings
        using offensive, defensive and win/loss metrics.
        """

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=42
        )
        model.fit(X, y)
        return model


    def run_conf_rank(self, conf_name, df_conf):
        """
        Perform walk-forward evaluation within a single conference:
        train on past years, predict the next year, generate ranking
        predictions and save the output files.
        """

        years = sorted(df_conf["year"].unique())
        results = {}

        for i in range(1, len(years)):
            train_years = years[:i]
            test_year = years[i]

            X_train, y_train, feat_cols = self.buildTrainingSet(train_years, df_conf)
            model = self.trainModel(X_train, y_train)

            X_test_full = self.buildFeaturesForYear(test_year, df_conf)
            if X_test_full is None:
                continue

            preds = model.predict(X_test_full[feat_cols])

            df_pred = pd.DataFrame({
                "tmID": X_test_full["tmID"],
                "year": test_year,
                "score": preds
            })

            df_pred = df_pred.sort_values("score").reset_index(drop=True)
            df_pred["rank_pred"] = df_pred["score"].rank(method="first").astype(int)

            real_rank = df_conf[df_conf["year"] == test_year][["tmID", "rank"]]
            df_pred = df_pred.merge(real_rank, on="tmID", how="left")

            folder = "teams_rank_predictions"
            os.makedirs(folder, exist_ok=True)
            out_path = f"{folder}/rank_predictions_{conf_name}_year_{test_year}.csv"
            df_pred.to_csv(out_path, index=False)

        return results


    def walkForwardRank(self):
        """
        Run the ranking process for every conference
        by applying walk-forward modeling individually to each one.
        """

        for conf in self.conferences:
            df_conf = self.conf_groups[conf]
            self.run_conf_rank(conf, df_conf)


def main():
        """
        Load the main team dataset, initialize TeamRank,
        and execute the full walk-forward ranking pipeline.
        """

        teams_df = pd.read_csv("../database/final/teams.csv")
        model = TeamRank(teams_df)
        model.walkForwardRank()


if __name__ == "__main__":
    main()
