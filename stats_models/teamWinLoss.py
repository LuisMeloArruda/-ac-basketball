import os
import numpy as np
import pandas as pd


class TeamWinLoss:
    def __init__(self, teams_df, N_years=3):
        self.N_years = N_years
        self.encoders = {}
        processed = self.preprocessTraining(teams_df)
        self.model = self.generateModel(processed["X"], processed["Y"])

    # ==========================================================
    # TRAINING PREPROCESSING
    # ==========================================================
    def preprocessTraining(self, df):

        from sklearn.preprocessing import LabelEncoder

        df = df.copy()
        df.columns = df.columns.str.strip()

        # Encode categorical IDs
        for col in ["tmID", "franchID", "confID", "arena"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        # ordering for temporal features
        df = df.sort_values(by=["tmID", "year"]).reset_index(drop=True)
        df["career_year"] = df.groupby("tmID").cumcount()

        # temporal win features
        df = self.computePrevWins(df)
        df = self.computeRollingWins(df, N=self.N_years)

        X = self.filterFeatures(df)
        Y = self.filterTargets(df)

        return {"X": X, "Y": Y, "df_full": df}

    # ==========================================================
    # INPUT PREPROCESSING (for a test year)
    # ==========================================================
    def preprocessInput(self, test_year, predictions_folder="players_predictions", teams_def_folder="teams_predictions"):

        # 1) load predicted player stats
        players_csv = os.path.join(predictions_folder, f"predictions_year_{test_year}.csv")
        if not os.path.exists(players_csv):
            raise FileNotFoundError(f"Missing players predictions: {players_csv}")

        pred_players = pd.read_csv(players_csv)
        if pred_players.empty:
            raise ValueError(f"Players prediction file is empty: {players_csv}")

        players_teams = pd.read_csv("../database/final/players_teams.csv")
        mapping = players_teams[players_teams["year"] == test_year][["playerID", "tmID"]]

        merged = pred_players.merge(mapping, on="playerID", how="left")

        # aggregate offense
        agg = merged.groupby(["tmID", "year"]).agg({
            "fgAttempted": "sum",
            "fgMade": "sum",
            "ftAttempted": "sum",
            "ftMade": "sum",
            "threeMade": "sum",
            "oRebounds": "sum",
            "dRebounds": "sum",
            "assists": "sum",
            "steals": "sum",
            "blocks": "sum",
            "turnovers": "sum",
            "PF": "sum",
            "points": "sum",
            "minutes": "sum"
        }).reset_index()

        # offense renaming
        agg["o_fgm"] = agg["fgMade"]
        agg["o_fga"] = agg["fgAttempted"]
        agg["o_ftm"] = agg["ftMade"]
        agg["o_fta"] = agg["ftAttempted"]
        agg["o_3pm"] = agg["threeMade"]
        agg["o_oreb"] = agg["oRebounds"]
        agg["o_dreb"] = agg["dRebounds"]
        agg["o_reb"] = agg["oRebounds"] + agg["dRebounds"]
        agg["o_asts"] = agg["assists"]
        agg["o_pf"] = agg["PF"]
        agg["o_stl"] = agg["steals"]
        agg["o_to"] = agg["turnovers"]
        agg["o_blk"] = agg["blocks"]
        agg["o_pts"] = agg["points"]

        # 2) merge team metadata
        teams_df = pd.read_csv("../database/final/teams.csv")
        teams_year = teams_df[teams_df["year"] == test_year][["tmID", "franchID", "confID", "arena"]]
        agg = agg.merge(teams_year, on="tmID", how="left")

        # 3) merge predicted defense or fallback to real defense
        teams_def_csv = os.path.join(teams_def_folder, f"defense_predictions_year_{test_year}.csv")
        if os.path.exists(teams_def_csv):
            def_df = pd.read_csv(teams_def_csv)
            agg = agg.merge(def_df, on=["tmID", "year"], how="left", suffixes=("", "_defpred"))
        else:
            real_def = teams_df[teams_df["year"] == test_year][["tmID"] + [c for c in teams_df.columns if c.startswith("d_")]]
            agg = agg.merge(real_def, on="tmID", how="left")

        # 4) re-encode categorical fields
        for col in ["franchID", "confID", "arena"]:
            agg[col] = agg[col].astype(str)
            if col in self.encoders:
                known = set(self.encoders[col].classes_)
                agg.loc[~agg[col].isin(known), col] = "-1"
                if "-1" not in self.encoders[col].classes_:
                    self.encoders[col].classes_ = np.append(self.encoders[col].classes_, "-1")
                agg[col] = self.encoders[col].transform(agg[col])

        # encode tmID
        agg["tmID"] = agg["tmID"].astype(str)
        if "tmID" in self.encoders:
            known = set(self.encoders["tmID"].classes_)
            agg.loc[~agg["tmID"].isin(known), "tmID"] = "-1"
            if "-1" not in self.encoders["tmID"].classes_:
                self.encoders["tmID"].classes_ = np.append(self.encoders["tmID"].classes_, "-1")
            agg["tmID"] = self.encoders["tmID"].transform(agg["tmID"])

        # 5) add historical win features
        past = teams_df[teams_df["year"] < test_year].copy()
        if not past.empty:
            for col in ["tmID", "franchID", "confID", "arena"]:
                past[col] = past[col].astype(str)
                if col in self.encoders:
                    known = set(self.encoders[col].classes_)
                    past.loc[~past[col].isin(known), col] = "-1"
                    if "-1" not in self.encoders[col].classes_:
                        self.encoders[col].classes_ = np.append(self.encoders[col].classes_, "-1")
                    past[col] = self.encoders[col].transform(past[col])

            past = past.sort_values(by=["tmID", "year"]).reset_index(drop=True)
            past["career_year"] = past.groupby("tmID").cumcount()

            past = self.computePrevWins(past)
            past = self.computeRollingWins(past, N=self.N_years)

            temporal_cols = [
                c for c in past.columns
                if c.startswith("prev_") or c.startswith(f"roll{self.N_years}_")
            ]

            past_latest = past.sort_values(["tmID", "year"]).groupby("tmID", as_index=False).last()
            keep_cols = ["tmID"] + temporal_cols
            keep_cols = [c for c in keep_cols if c in past_latest.columns]

            past_latest = past_latest[keep_cols]
            agg = agg.merge(past_latest, on="tmID", how="left")
        else:
            agg["prev_won"] = np.nan
            agg[f"roll{self.N_years}_won"] = np.nan

        # final feature selection
        X_test = agg[self.filterFeatures(agg).columns]

        # fill missing numeric with zero (safe fallback)
        X_test = X_test.fillna(0)

        return X_test

    # ==========================================================
    # TEMPORAL FEATURES: prev_won and rollN_won
    # ==========================================================
    def computePrevWins(self, df):
        df = df.sort_values(by=["tmID", "career_year"])
        df["prev_won"] = df.groupby("tmID")["won"].shift(1)
        return df

    def computeRollingWins(self, df, N=3):
        df = df.sort_values(by=["tmID", "career_year"])
        df[f"roll{N}_won"] = (
            df.groupby("tmID")["won"]
            .rolling(N, min_periods=1)
            .mean()
            .groupby(level=0)
            .shift(1)
            .values
        )
        return df

    # ==========================================================
    # FEATURE / TARGET SELECTION
    # ==========================================================
    def filterFeatures(self, df):
        base = [
            "tmID", "franchID", "confID", "year", "arena",
            "o_fgm", "o_fga", "o_ftm", "o_fta", "o_3pm",
            "o_oreb", "o_dreb", "o_reb",
            "o_asts", "o_pf", "o_stl", "o_to", "o_blk", "o_pts"
        ]

        def_cols = [c for c in df.columns if c.startswith("d_")]
        temporal = [c for c in df.columns if c in ["prev_won", f"roll{self.N_years}_won"]]

        feature_cols = base + def_cols + temporal
        feature_cols = [c for c in feature_cols if c in df.columns]

        return df[feature_cols]

    def filterTargets(self, df):
        return df[["won", "lost"]]

    # ==========================================================
    # MODEL TRAINING
    # ==========================================================
    def generateModel(self, X, Y):

        from catboost import CatBoostRegressor
        from sklearn.multioutput import MultiOutputRegressor

        model = MultiOutputRegressor(
            CatBoostRegressor(
                iterations=700,
                depth=6,
                learning_rate=0.03,
                loss_function="RMSE",
                verbose=False
            )
        )
        model.fit(X, Y)
        return model

    # ==========================================================
    # WALK-FORWARD VALIDATION
    # ==========================================================
    def walkForwardWL(self, teams_df, predictions_folder="players_predictions", teams_def_folder="teams_predictions"):

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        years = sorted(teams_df["year"].unique())
        results = {}

        for i in range(1, len(years)):
            test_year = years[i]
            train_years = years[:i]

            train_df = teams_df[teams_df["year"].isin(train_years)].copy()
            processed = self.preprocessTraining(train_df)
            X_train, Y_train = processed["X"], processed["Y"]

            self.model = self.generateModel(X_train, Y_train)

            # save debug info
            df_debug = processed["df_full"].copy()
            try:
                df_debug["tmID_raw"] = self.encoders["tmID"].inverse_transform(df_debug["tmID"].astype(int))
            except:
                df_debug["tmID_raw"] = df_debug["tmID"]

            feature_cols = X_train.columns.tolist()
            sample = df_debug[["tmID_raw"] + feature_cols]

            os.makedirs("teams_wl_debug", exist_ok=True)
            debug_path = f"teams_wl_debug/debug_features_year_{test_year}.txt"
            sample.to_string(open(debug_path, "w"), index=False)
            print(f"[DEBUG] Training features saved to {debug_path}")

            # build test input
            try:
                X_test = self.preprocessInput(
                    test_year,
                    predictions_folder=predictions_folder,
                    teams_def_folder=teams_def_folder
                )
            except Exception as e:
                print(f"[WARNING] Could not build input for year {test_year}: {e}")
                continue

            preds = self.model.predict(X_test)

            # save predictions
            pred_df = pd.DataFrame(preds, columns=["won", "lost"])
            try:
                pred_df.insert(0, "tmID", self.encoders["tmID"].inverse_transform(X_test["tmID"].astype(int)))
            except:
                pred_df.insert(0, "tmID", X_test["tmID"].values)
            pred_df.insert(1, "year", test_year)

            os.makedirs("teams_wl_predictions", exist_ok=True)
            out_path = f"teams_wl_predictions/wl_predictions_year_{test_year}.csv"
            pred_df.to_csv(out_path, index=False)
            print(f"[DEBUG] WL predictions saved to {out_path}")

            # compute metrics
            real_df = teams_df[teams_df["year"] == test_year].copy().sort_values("tmID").reset_index(drop=True)
            pred_sorted = pred_df.sort_values("tmID").reset_index(drop=True)

            if real_df.empty or pred_sorted.empty:
                print(f"[WARNING] Empty real or predicted dataset for {test_year}, skipping.")
                continue

            y_true = real_df[["won", "lost"]].values
            y_pred = pred_sorted[["won", "lost"]].values

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            results[test_year] = {"mae": mae, "mse": mse, "r2": r2}
            print(f"[OK] Year {test_year}: MAE={mae:.2f} MSE={mse:.2f} R²={r2:.3f}")

        print("\n===== WIN/LOSS YEARLY METRICS =====")
        for y, m in results.items():
            print(f"Year {y}: MAE={m['mae']:.2f} | MSE={m['mse']:.2f} | R²={m['r2']:.3f}")

        return results


def main():

    teams_df = pd.read_csv("../database/final/teams.csv")
    model = TeamWinLoss(teams_df, N_years=3)
    model.walkForwardWL(teams_df, predictions_folder="players_predictions", teams_def_folder="teams_predictions")


if __name__ == "__main__":
    main()
