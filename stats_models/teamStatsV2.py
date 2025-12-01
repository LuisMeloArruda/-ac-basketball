import pandas as pd
import numpy as np


class TeamStats:
    def __init__(self, teams_df):
        """
        Initialize the TeamStats model:
        - Preprocess the input team dataframe
        - Encode categorical columns
        - Generate defensive temporal features
        - Train initial defensive model
        """
        self.encoders = {}
        processed = self.preprocessTraining(teams_df)
        self.model = self.generateModel(processed["X"], processed["Y"])

    # ==========================================================
    # TRAINING PREPROCESSING
    # ==========================================================
    def preprocessTraining(self, df):
        """
        Full preprocessing pipeline for training:
        - Encode categorical identifiers (team, franchise, conference, arena, name)
        - Sort teams by season and assign career_year sequence
        - Compute previous-year and rolling defensive features
        - Select model features (X) and defensive targets (Y)
        - Return processed data components
        """

        from sklearn.preprocessing import LabelEncoder

        df = df.copy()

        # Encode categories
        for col in ["tmID", "franchID", "confID", "arena", "name"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        df = df.sort_values(by=["tmID", "year"]).reset_index(drop=True)
        df["career_year"] = df.groupby("tmID").cumcount()

        # Compute temporal features
        df = self.computePrevDefense(df)
        df = self.computeRollingDefense(df, N=3)

        X = self.filterFeatures(df)
        Y = self.filterTargets(df)

        return {"X": X, "Y": Y, "df_full": df}

    # ==========================================================
    # INPUT PREPROCESSING (PREDICTED PLAYER OFFENSE)
    # ==========================================================
    def preprocessInput(self, predicted_players_df):
        """
        Preprocess predicted player stats to build team-level OFFENSIVE input:
        - Load historical player-team mapping for the same season
        - Merge predicted stats with team assignments
        - Aggregate all player stats per team
        - Merge team metadata (franchID, confID, arena, name)
        - Encode all categorical team attributes using fitted encoders
        - Merge defensive temporal history (prev and rolling features)
        - Return fully prepared feature matrix for inference
        """

        players_teams = pd.read_csv("../database/final/players_teams.csv")
        test_year = predicted_players_df["year"].iloc[0]

        players_year = players_teams[players_teams["year"] == test_year][["playerID", "tmID"]]
        merged = predicted_players_df.merge(players_year, on="playerID", how="left")

        # Aggregate player OFFENSE → team offense
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

        # Normalize offense column names
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

        # Merge team metadata
        teams_df = pd.read_csv("../database/final/teams.csv")
        teams_year = teams_df[teams_df["year"] == test_year][["tmID", "franchID", "confID", "arena", "name"]]

        agg = agg.merge(teams_year, on="tmID", how="left")

        # Encode categorical team attributes
        for col in ["franchID", "confID", "arena", "name"]:
            agg[col] = agg[col].astype(str)
            known = set(self.encoders[col].classes_)
            agg.loc[~agg[col].isin(known), col] = "-1"

            if "-1" not in self.encoders[col].classes_:
                self.encoders[col].classes_ = np.append(self.encoders[col].classes_, "-1")

            agg[col] = self.encoders[col].transform(agg[col])

        # Encode tmID
        agg["tmID"] = agg["tmID"].astype(str)
        known = set(self.encoders["tmID"].classes_)
        agg.loc[~agg["tmID"].isin(known), "tmID"] = "-1"

        if "-1" not in self.encoders["tmID"].classes_:
            self.encoders["tmID"].classes_ = np.append(self.encoders["tmID"].classes_, "-1")

        agg["tmID"] = self.encoders["tmID"].transform(agg["tmID"])

        # Load all historical team seasons
        teams_full = pd.read_csv("../database/final/teams.csv")
        past = teams_full[teams_full["year"] < test_year].copy()

        if not past.empty:
            # Encode past categorical data
            for col in ["tmID", "franchID", "confID", "arena", "name"]:
                past[col] = past[col].astype(str)
                known = set(self.encoders[col].classes_)
                past.loc[~past[col].isin(known), col] = "-1"

                if "-1" not in self.encoders[col].classes_:
                    self.encoders[col].classes_ = np.append(self.encoders[col].classes_, "-1")

                past[col] = self.encoders[col].transform(past[col])

            past = past.sort_values(by=["tmID", "year"]).reset_index(drop=True)
            past["career_year"] = past.groupby("tmID").cumcount()

            # Compute past defensive temporal features
            past = self.computePrevDefense(past)
            past = self.computeRollingDefense(past, N=3)

            # Only keep latest defensive history per team
            temporal_cols = [c for c in past.columns if c.startswith("prev_") or c.startswith("roll3_")]
            past_latest = past.sort_values(["tmID", "year"]).groupby("tmID", as_index=False).last()
            keep_cols = ["tmID"] + [c for c in temporal_cols if c in past_latest.columns]
            past_latest = past_latest[keep_cols]

            agg = agg.merge(past_latest, on="tmID", how="left")

        else:
            # If no history exists, fill temporal features with NaN
            temp_example = [
                "d_fga", "d_ftm", "d_3pm", "d_3pa",
                "d_oreb", "d_dreb", "d_asts",
                "d_pf", "d_stl", "d_to", "d_blk", "d_pts"
            ]
            for col in temp_example:
                agg[f"prev_{col}"] = np.nan
                agg[f"roll3_{col}"] = np.nan

        return agg[self.filterFeatures(agg).columns]

    # ==========================================================
    # TEMPORAL FEATURE ENGINEERING FOR TEAM DEFENSE
    # ==========================================================
    def computePrevDefense(self, df):
        """
        Compute previous-season defensive statistics for each team.
        Creates prev_X columns using career_year ordering.
        """

        defensive_cols = [
            "d_fga", "d_ftm", "d_3pm", "d_3pa",
            "d_oreb", "d_dreb", "d_asts",
            "d_pf", "d_stl", "d_to", "d_blk", "d_pts"
        ]

        df = df.sort_values(by=["tmID", "career_year"])

        for col in defensive_cols:
            df[f"prev_{col}"] = df.groupby("tmID")[col].shift(1)

        return df

    def computeRollingDefense(self, df, N=3):
        """
        Compute rolling N-year defensive averages for each team.
        Uses a shift(1) to avoid leaking the current-season stats.
        """

        defensive_cols = [
            "d_fga", "d_ftm", "d_3pm", "d_3pa",
            "d_oreb", "d_dreb", "d_asts",
            "d_pf", "d_stl", "d_to", "d_blk", "d_pts"
        ]

        df = df.sort_values(by=["tmID", "career_year"])

        for col in defensive_cols:
            roll_col = f"roll{N}_{col}"
            df[roll_col] = (
                df.groupby("tmID")[col]
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
        """
        Select team-level model input features:
        - Team identifiers and encoded metadata
        - Offensive totals aggregated from predicted players
        - Previous/rolling defensive features
        """
        feature_cols = [
            "tmID", "franchID", "confID", "year", "arena", "name",
            "o_fgm", "o_fga", "o_ftm", "o_fta", "o_3pm",
            "o_oreb", "o_dreb", "o_reb",
            "o_asts", "o_pf", "o_stl", "o_to", "o_blk", "o_pts"
        ]

        feature_cols += [c for c in df.columns if c.startswith("prev_")]
        feature_cols += [c for c in df.columns if c.startswith("roll3_")]

        return df[feature_cols]

    def filterTargets(self, df):
        """
        Select defensive statistical targets to predict for each team.
        """
        target_cols = [
            "d_fga", "d_ftm", "d_3pm", "d_3pa",
            "d_oreb", "d_dreb", "d_asts",
            "d_pf", "d_stl", "d_to", "d_blk", "d_pts"
        ]
        return df[target_cols]

    # ==========================================================
    # MODEL TRAINING
    # ==========================================================
    def generateModel(self, X, Y):
        """
        Train a MultiOutputRegressor using CatBoostRegressor
        to simultaneously predict all defensive statistics.
        """

        from catboost import CatBoostRegressor
        from sklearn.multioutput import MultiOutputRegressor

        model = MultiOutputRegressor(
            CatBoostRegressor(
                iterations=700,
                depth=8,
                learning_rate=0.03,
                loss_function="RMSE",
                verbose=False,
            )
        )
        model.fit(X, Y)
        return model

    # ==========================================================
    # WALK-FORWARD VALIDATION FOR TEAM DEFENSIVE MODEL
    # ==========================================================
    def walkForwardTeams(self, teams_df, predictions_folder="players_predictions"):
        """
        Perform walk-forward validation of the team defensive model:
        - Train model on seasons before test_year
        - Load predicted player statistics for test_year
        - Aggregate player predictions into team offense inputs
        - Predict team defensive performance
        - Save predictions and debug files
        - Compute MAE, MSE, and R² for each year
        """

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import os

        years = sorted(teams_df["year"].unique())
        results_by_year = {}

        for i in range(1, len(years)):

            test_year = years[i]
            train_years = years[:i]

            # ------------------------------
            # 1) TRAIN TEAM MODEL
            # ------------------------------
            train_df = teams_df[teams_df["year"].isin(train_years)].copy()
            processed = self.preprocessTraining(train_df)

            X_train, Y_train = processed["X"], processed["Y"]
            self.model = self.generateModel(X_train, Y_train)

            # Save debug features
            df_debug = processed["df_full"].copy()

            try:
                df_debug["tmID_raw"] = self.encoders["tmID"].inverse_transform(
                    df_debug["tmID"].astype(int)
                )
            except:
                df_debug["tmID_raw"] = df_debug["tmID"]

            feature_cols = X_train.columns.tolist()
            sample = df_debug[["tmID_raw"] + feature_cols]

            os.makedirs("teams_debug", exist_ok=True)
            debug_path = f"teams_debug/debug_features_year_{test_year}.txt"
            sample.to_string(open(debug_path, "w"), index=False)
            print(f"[DEBUG] Training features saved to {debug_path}")

            # ------------------------------
            # 2) LOAD PLAYER PREDICTED STATS
            # ------------------------------
            csv_path = os.path.join(predictions_folder, f"predictions_year_{test_year}.csv")

            if not os.path.exists(csv_path):
                print(f"[WARNING] Missing predictions file: {csv_path} — skipping year.")
                continue

            pred_players = pd.read_csv(csv_path)

            if pred_players.empty:
                print(f"[WARNING] Empty prediction file: {csv_path} — skipping year.")
                continue

            # ------------------------------
            # 3) AGGREGATE PLAYER → TEAM OFFENSE
            # ------------------------------
            agg_offense = self.preprocessInput(pred_players)

            # Save offensive aggregation
            agg_offense_to_save = agg_offense.copy()
            try:
                agg_offense_to_save["tmID"] = self.encoders["tmID"].inverse_transform(
                    agg_offense_to_save["tmID"].astype(int)
                )
            except Exception as e:
                print("[WARN] Could not decode tmID for offense:", e)

            os.makedirs("teams_offense_predictions", exist_ok=True)
            off_path = f"teams_offense_predictions/offense_predictions_year_{test_year}.csv"
            agg_offense_to_save.to_csv(off_path, index=False)
            print(f"[DEBUG] Team offense saved to {off_path}")

            X_test = agg_offense

            # ------------------------------
            # 4) PREDICT TEAM DEFENSE
            # ------------------------------
            preds = self.model.predict(X_test)

            pred_df = pd.DataFrame(preds, columns=self.filterTargets(train_df).columns)
            pred_df.insert(0, "tmID", X_test["tmID"].values)
            pred_df.insert(1, "year", test_year)

            try:
                pred_df["tmID"] = self.encoders["tmID"].inverse_transform(
                    pred_df["tmID"].astype(int)
                )
            except:
                pass

            os.makedirs("teams_defense_predictions", exist_ok=True)
            out_path = f"teams_predictions/defense_predictions_year_{test_year}.csv"
            pred_df.to_csv(out_path, index=False)
            print(f"[DEBUG] Team defense predictions saved to {out_path}")

            # ------------------------------
            # 5) COMPUTE METRICS
            # ------------------------------
            real_df = teams_df[teams_df["year"] == test_year].copy()
            real_df = real_df.sort_values(by="tmID").reset_index(drop=True)

            pred_sorted = pred_df.sort_values(by="tmID").reset_index(drop=True)

            y_true = real_df[self.filterTargets(real_df).columns].values
            y_pred = pred_sorted[self.filterTargets(real_df).columns].values

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            results_by_year[test_year] = {"mae": mae, "mse": mse, "r2": r2}

            print(f"[OK] Year {test_year} → MAE={mae:.2f}, MSE={mse:.2f}, R²= {r2:.3f}")

        # ------------------------------
        # SUMMARY
        # ------------------------------
        print("\n===== TEAM DEFENSE YEARLY METRICS =====")
        for y, m in results_by_year.items():
            print(f"Year {y}: MAE={m['mae']:.2f} | MSE={m['mse']:.2f} | R²= {m['r2']:.3f}")


def main():
    """
    Load historical team dataset, create the TeamStats model,
    and run walk-forward validation using player prediction CSVs.
    """

    teams_df = pd.read_csv("../database/final/teams.csv")
    model = TeamStats(teams_df)

    model.walkForwardTeams(
        teams_df,
        predictions_folder="players_predictions"
    )

if __name__ == "__main__":
    main()
