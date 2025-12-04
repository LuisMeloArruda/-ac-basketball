import pandas as pd
import numpy as np
import os

class PlayerStats:
    def __init__(self, players_team_df, player_df, N_years=3):
        self.players_team_df = players_team_df
        self.player_df = player_df
        self.N_years = N_years
        self.encoders = {}
        self.encoded = ["playerID", "tmID", "pos", "college"]

        self.target_columns = [
            "minutes", "points", "oRebounds", "dRebounds",
            "assists", "steals", "blocks", "turnovers", "PF",
            "fgAttempted", "fgMade", "ftAttempted", "ftMade", "threeMade"
        ]

        # deal with rookies
        df_sorted = players_team_df.sort_values(by=["playerID", "year"])
        rookies = df_sorted.groupby("playerID").first()
        self.rookie_baseline = rookies[self.target_columns].mean()


    # ==========================================================
    # TRAINING PREPROCESSING (ORQUESTRADOR)
    # ==========================================================
    def preprocessTraining(self, df):
        
        df = self.mergeBio(df)

        df = self.computeFeatures(df)

        df = self.encodeCategoricals(df, fit=True)

        X = self.filterFeatures(df)
        Y = self.filterTargets(df)

        return {"X": X, "Y": Y, "df_full": df}

    # ==========================================================
    # FUTURE INPUT PREPROCESSING
    # ==========================================================
    def preprocessInput(self, df_future, df_history):
        
        df = pd.concat([df_history.copy(), df_future.copy()], ignore_index=True)

        df = self.mergeBio(df)
        df = self.computeFeatures(df)

        future_year = df_future["year"].iloc[0]
        df_future_processed = df[df["year"] == future_year].copy()

        df_future_processed = self.encodeCategoricals(df_future_processed, fit=False)

        # 5. Selecionar colunas finais
        X = self.filterFeatures(df_future_processed)
        
        return X

    # ==========================================================
    # MERGE BIO DATA
    # ==========================================================
    def mergeBio(self, df):
        df = df.merge(
            self.player_df[["bioID", "birthDate", "pos", "height", "weight", "college"]],
            left_on="playerID",
            right_on="bioID",
            how="left"
        )
        df.drop(columns=["bioID"], inplace=True, errors="ignore")
        return df

    # ==========================================================
    # FEATURE ENGINEERING
    # ==========================================================
    def computeFeatures(self, df):
        df = self.computeAge(df)
        df = df.sort_values(by=["playerID", "year"]).reset_index(drop=True)

        df["career_year"] = df.groupby("playerID").cumcount()

        df = self.computePrevStats(df)
        df = self.computeRollingStats(df)
        
        return df

    # ==========================================================
    # ENCODING
    # ==========================================================
    def encodeCategoricals(self, df, fit=True):
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        cols_to_fill = ["pos", "college"] 
        all_cats = list(set(cols_to_fill + self.encoded))

        for col in all_cats:
            if col in df.columns:
                df[col] = df[col].fillna("-1").astype(str)

        for col in self.encoded:
            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            
            else:
                if col in self.encoders:
                    enc = self.encoders[col]
                    
                    known_classes = set(enc.classes_)
                    df.loc[~df[col].isin(known_classes), col] = "-1"

                    if "-1" not in enc.classes_:
                        enc.classes_ = np.append(enc.classes_, "-1")
                    
                    df[col] = enc.transform(df[col])
        
        return df

    # ==========================================================
    # TEMPORAL FEATURE ENGINEERING
    # ==========================================================
    def computeAge(self, df, current_year=2025):

        df["birthDate"] = pd.to_datetime(df["birthDate"], errors='coerce')
        df["birthYear"] = df["birthDate"].dt.year

        year_max = df["year"].max()
        df["age"] = current_year - (year_max - df["year"]) - df["birthYear"]

        df.loc[df["birthYear"].isna(), "age"] = np.nan
        df["age_sq"] = df["age"] ** 2

        return df

    def computePrevStats(self, df):

        df = df.sort_values(by=["playerID", "career_year"])

        for col in self.target_columns:
            prev_col = f"prev_{col}"

            df[prev_col] = df.groupby("playerID")[col].shift(1)

            df.loc[df["career_year"] == 0, prev_col] = self.rookie_baseline[col]

        return df


    def computeRollingStats(self, df):

        df = df.sort_values(by=["playerID", "career_year"])

        for col in self.target_columns:
            roll_col = f"roll{self.N_years}_{col}"

            df[roll_col] = (
                df.groupby("playerID")[col]
                .rolling(self.N_years, min_periods=1)
                .mean()
                .groupby(level=0)
                .shift(1)
                .values
            )

            df.loc[df["career_year"] == 0, roll_col] = self.rookie_baseline[col]

        return df

    # ==========================================================
    # FEATURE / TARGET SELECTION
    # ==========================================================
    def filterFeatures(self, df):

        feature_cols = [
            "playerID", "tmID", "age", "age_sq",
            "pos", "college", "height", "weight"
        ]

        feature_cols += [c for c in df.columns if c.startswith("prev_")]
        feature_cols += [c for c in df.columns if c.startswith("roll")]

        return df[feature_cols]

    def filterTargets(self, df):

        return df[self.target_columns]

    # ==========================================================
    # MODEL TRAINING
    # ==========================================================
    def generateModel(self, training_dict):

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
    def walkForward(self):

        results_by_year = {}
        years = sorted(self.players_team_df["year"].unique())

        for i in range(9, len(years)):
            train_years = years[:i]
            test_year = years[i]

            train_df = self.players_team_df[self.players_team_df["year"].isin(train_years)].copy()
            train_processed = self.preprocessTraining(train_df)
            X_train = train_processed["X"]
            Y_train = train_processed["Y"]

            self.model = self.generateModel(train_processed)

            df_debug = train_processed["df_full"].copy()

            feature_cols = X_train.columns.tolist()

            sample = df_debug[["playerID", "career_year", "tmID"] + feature_cols]

            os.makedirs("players_debug", exist_ok=True)
            filename = f"players_debug/debug_features_year_{test_year}.txt"
            sample.to_string(open(filename, "w"), index=False)
            print(f"[DEBUG] Training features saved to {filename}")

            test_df = self.players_team_df[self.players_team_df["year"] == test_year].copy()
            history_df = self.players_team_df[self.players_team_df["year"] < test_year]

            X_test = self.preprocessInput(test_df, history_df)

            preds = self.model.predict(X_test)
            real = self.filterTargets(test_df).values

            test_df_copy = test_df.copy()

            try:
                test_df_copy["playerID_raw"] = self.encoders["playerID"].inverse_transform(
                    test_df_copy["playerID"].astype(int)
                )
            except:
                test_df_copy["playerID_raw"] = test_df_copy["playerID"]

            try:
                test_df_copy["tmID_raw"] = self.encoders["tmID"].inverse_transform(
                    test_df_copy["tmID"].astype(int)
                )
            except:
                test_df_copy["tmID_raw"] = test_df_copy["tmID"]

            pred_df = pd.DataFrame(preds, columns=self.target_columns)

            pred_df.insert(0, "playerID", test_df_copy["playerID_raw"].values)
            pred_df.insert(1, "year", test_year)
            pred_df.insert(2, "tmID", test_df_copy["tmID_raw"].values)

            os.makedirs("players_predictions", exist_ok=True)
            pred_df.to_csv(f"players_predictions/predictions_year_{test_year}.csv", index=False)

            print(f"[DEBUG] Predictions saved to predictions_year_{test_year}.csv")
            print(f"[OK] Year {test_year} evaluated (trained up to {train_years[-1]}).")


def main():
    players_teams = pd.read_csv("../database/final/players_teams.csv")
    players_info = pd.read_csv("../database/final/players.csv")

    model = PlayerStats(players_teams, players_info)
    model.walkForward()

if __name__ == "__main__":
    main()
