import pandas as pd
import numpy as np
import os

class PlayerStats:
    def __init__(self, training_df, player_info_df, N_years=3):
        """
        Initialize PlayerStats:
        - Merge player metadata into the training dataframe
        - Store encoders and window size (N_years)
        - Preprocess training data and fit the initial model
        """

        self.N_years = N_years
        self.encoders = {}
        self.player_info_df = player_info_df

        training_df = self.preprocessTraining(training_df)
        self.model = self.generateModel(training_df)

        self.target_columns = [
            "minutes", "points", "oRebounds", "dRebounds",
            "assists", "steals", "blocks", "turnovers", "PF",
            "fgAttempted", "fgMade", "ftAttempted", "ftMade", "threeMade"
        ]

    # ==========================================================
    # TRAINING PREPROCESSING
    # ==========================================================
    def preprocessTraining(self, df):
        """
        Full training preprocessing pipeline:
        - Merge player metadata
        - Compute age, previous-year stats, and rolling averages
        - Encode categorical fields
        - Select and return X (features), Y (targets), and df_full
        """

        df = df.merge(
            self.player_info_df[["bioID", "birthDate", "pos", "height", "weight", "college"]],
            left_on="playerID",
            right_on="bioID",
            how="left"
        )
        df.drop(columns=["bioID"], inplace=True, errors="ignore")

        df = self.computeAge(df)
        df = df.sort_values(by=["playerID", "year"]).reset_index(drop=True)

        df["career_year"] = df.groupby("playerID").cumcount()

        df = self.computePrevStats(df)
        df = self.computeRollingStats(df)

        for col in ["pos", "college"]:
            df[col] = df[col].fillna("-1").astype(str)

        from sklearn.preprocessing import LabelEncoder
        for col in ["playerID", "tmID", "pos", "college"]:
            self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col].astype(str))

        useless_cols = ["stint", "lgID", "birthDate", "birthYear"]
        df.drop(columns=[c for c in useless_cols if c in df.columns], inplace=True)

        X = self.filterFeatures(df)
        Y = self.filterTargets(df)

        return {"X": X, "Y": Y, "df_full": df}

    # ==========================================================
    # FUTURE INPUT PREPROCESSING
    # ==========================================================
    def preprocessInput(self, df_future, df_history):
        """
        Preprocess data for a future (unseen) year:
        - Combine history + future
        - Recompute temporal features using full career sequence
        - Select rows only for the new year
        - Apply encoders consistently (handling unknown values)
        """

        df = pd.concat([df_history.copy(), df_future.copy()], ignore_index=True)

        df = df.merge(
            self.player_info_df[["bioID", "birthDate", "pos", "height", "weight", "college"]],
            left_on="playerID",
            right_on="bioID",
            how="left"
        )

        df.drop(columns=["bioID"], inplace=True, errors="ignore")

        df = self.computeAge(df)
        df = df.sort_values(by=["playerID", "year"]).reset_index(drop=True)

        df["career_year"] = df.groupby("playerID").cumcount()

        df = self.computePrevStats(df)
        df = self.computeRollingStats(df)

        df_future_processed = df[df["year"] == df_future["year"].iloc[0]].copy()

        for col in ["playerID", "tmID", "pos", "college"]:
            df_future_processed[col] = df_future_processed[col].astype(str)

            known = set(self.encoders[col].classes_)
            df_future_processed.loc[~df_future_processed[col].isin(known), col] = "-1"

            if "-1" not in self.encoders[col].classes_:
                self.encoders[col].classes_ = np.append(self.encoders[col].classes_, "-1")

            df_future_processed[col] = self.encoders[col].transform(df_future_processed[col])

        X = self.filterFeatures(df_future_processed)
        return X

    # ==========================================================
    # TEMPORAL FEATURE ENGINEERING
    # ==========================================================
    def computeAge(self, df, current_year=2025):
        """
        Compute player age and age squared for each season:
        age = (estimated birth year) relative to current season alignment.
        Missing birthdates result in NaN ages.
        """
        df["birthDate"] = pd.to_datetime(df["birthDate"], errors='coerce')
        df["birthYear"] = df["birthDate"].dt.year

        year_max = df["year"].max()
        df["age"] = current_year - (year_max - df["year"]) - df["birthYear"]

        df.loc[df["birthYear"].isna(), "age"] = np.nan
        df["age_sq"] = df["age"] ** 2

        return df

    def computePrevStats(self, df):
        """
        Create previous-season statistics for each player using career ordering.
        prev_X contains the stat from the immediately preceding year.
        """
        stat_columns = [
            "minutes", "points", "oRebounds", "dRebounds",
            "assists", "steals", "blocks", "turnovers", "PF",
            "fgAttempted", "fgMade", "ftAttempted", "ftMade", "threeMade"
        ]

        df = df.sort_values(by=["playerID", "career_year"])

        for col in stat_columns:
            df[f"prev_{col}"] = df.groupby("playerID")[col].shift(1)

        return df

    def computeRollingStats(self, df):
        """
        Compute rolling N-year averages for each stat.
        A shift(1) is applied to avoid leakage from the current year.
        """
        stat_columns = [
            "minutes", "points", "oRebounds", "dRebounds",
            "assists", "steals", "blocks", "turnovers", "PF",
            "fgAttempted", "fgMade", "ftAttempted", "ftMade", "threeMade"
        ]

        df = df.sort_values(by=["playerID", "career_year"])

        for col in stat_columns:
            rolling_col = f"roll{self.N_years}_{col}"
            df[rolling_col] = (
                df.groupby("playerID")[col]
                .rolling(self.N_years, min_periods=1)
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
        Select model input features, including:
        - Encoded identifiers
        - Demographic information (age, height, weight)
        - Previous-season stats
        - Rolling-window features
        """

        feature_cols = [
            "playerID", "tmID", "age", "age_sq",
            "pos", "college", "height", "weight"
        ]

        feature_cols += [c for c in df.columns if c.startswith("prev_")]
        feature_cols += [c for c in df.columns if c.startswith("roll")]
        feature_cols = [c for c in feature_cols if c in df.columns]

        return df[feature_cols]

    def filterTargets(self, df):
        """
        Select the target statistical categories for prediction.
        """

        target_cols = [
            "minutes", "points", "oRebounds", "dRebounds",
            "assists", "steals", "blocks", "turnovers", "PF",
            "fgAttempted", "fgMade", "ftAttempted", "ftMade", "threeMade"
        ]
        return df[target_cols]

    # ==========================================================
    # MODEL TRAINING
    # ==========================================================
    def generateModel(self, training_dict):
        """
        Train a MultiOutputRegressor using a CatBoostRegressor backend.
        Handles multiple statistical targets simultaneously.
        """
        from catboost import CatBoostRegressor
        from sklearn.multioutput import MultiOutputRegressor

        X = training_dict["X"]
        Y = training_dict["Y"]

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
    # WALK-FORWARD VALIDATION
    # ==========================================================
    def walkForward(self, df_raw):
        """
        Perform walk-forward validation:
        - Train on all years up to year N
        - Predict for year N+1
        - Save predictions and debug files
        - Compute and print metrics (MAE, MSE, R²)
        """

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        results_by_year = {}
        years = sorted(df_raw["year"].unique())

        for i in range(1, len(years)):
            train_years = years[:i]
            test_year = years[i]

            train_df = df_raw[df_raw["year"].isin(train_years)].copy()
            train_processed = self.preprocessTraining(train_df)
            X_train = train_processed["X"]
            Y_train = train_processed["Y"]

            self.model = self.generateModel(train_processed)

            df_debug = train_processed["df_full"].copy()
            df_debug["playerID_raw"] = self.encoders["playerID"].inverse_transform(
                df_debug["playerID"].astype(int)
            )

            feature_cols = X_train.columns.tolist()
            sample = df_debug[["playerID_raw"] + ["career_year"] + feature_cols]
            os.makedirs("players_debug", exist_ok=True)
            filename = f"players_debug/debug_features_year_{test_year}.txt"
            sample.to_string(open(filename, "w"), index=False)
            print(f"[DEBUG] Training features saved to {filename}")

            test_df = df_raw[df_raw["year"] == test_year].copy()
            history_df = df_raw[df_raw["year"] < test_year]
            X_test = self.preprocessInput(test_df, history_df)

            preds = self.model.predict(X_test)
            real = self.filterTargets(test_df).values

            test_df_copy = test_df.copy()
            test_df_copy["playerID_raw"] = test_df_copy["playerID"]

            try:
                test_df_copy["playerID_raw"] = self.encoders["playerID"].inverse_transform(
                    test_df_copy["playerID"].astype(int)
                )
            except:
                pass

            pred_df = pd.DataFrame(preds, columns=self.target_columns)
            pred_df.insert(0, "playerID", test_df_copy["playerID_raw"].values)
            pred_df.insert(1, "year", test_year)
            os.makedirs("players_predictions", exist_ok=True)
            pred_df.to_csv(f"players_predictions/predictions_year_{test_year}.csv", index=False)
            print(f"[DEBUG] Predictions saved to predictions_year_{test_year}.csv")

            mae = mean_absolute_error(real, preds)
            mse = mean_squared_error(real, preds)
            r2 = r2_score(real, preds)

            results_by_year[test_year] = {"mae": mae, "mse": mse, "r2": r2}
            print(f"[OK] Year {test_year} evaluated (trained up to {train_years[-1]}).")

        print("\n===== YEARLY METRICS =====\n")
        for year, metrics in results_by_year.items():
            print(f"Year {year}:  MAE={metrics['mae']:.2f}   MSE={metrics['mse']:.2f}   R²={metrics['r2']:.3f}")


def main():
    players_teams = pd.read_csv("../database/final/players_teams.csv")
    players_info = pd.read_csv("../database/final/players.csv")

    model = PlayerStats(training_df=players_teams, player_info_df=players_info, N_years=3)
    model.walkForward(players_teams)

if __name__ == "__main__":
    main()
