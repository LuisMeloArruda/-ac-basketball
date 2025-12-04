import pandas as pd
import numpy as np
import os

class CoachesModel:
    def __init__(self, coaches_df, teams_df):
        self.coaches_df = coaches_df.copy()
        self.teams_df = teams_df.copy()

        self.full_df = self.merge(self.coaches_df, self.teams_df)

        self.encoded = ["tmID", "lgID", "coachID"]
        self.encoders = self.buildEncoders()

    # ==========================================================
    # FEATURE SELECTION
    # ==========================================================
    def filterFeatures(self, df):
        feature_cols = [
            "prev_coach_win_pct", "prev_coach_made_playoffs", "coach_tenure",
        ]

        return df[feature_cols]

    # ==========================================================
    # TARGET SELECTION
    # ==========================================================
    def filterTarget(self, df):
        return df["stint"]

    # ==========================================================
    # TRAINING PREPROCESSING
    # ==========================================================
    def preprocessTraining(self, full_df):
        df = self.computeCoachPrevWin(full_df)
        df = self.computePrevCoachMadePlayoffs(df)
        df = self.computeCoachTenure(df)

        df = self.encodeCategoricals(df)

        x = self.filterFeatures(df)
        y = self.filterTarget(df)

        return x, y, df

    # ==========================================================
    # TEST INPUT PREPROCESSING
    # ==========================================================
    def preprocessInput(self, full_df_test):
        df = self.computeCoachPrevWin(full_df_test)
        df = self.computePrevCoachMadePlayoffs(df)
        df = self.computeCoachTenure(df)

        df = self.decodeCategoricals(df)

        x = self.filterFeatures(df)
        return x, df

    # ==========================================================
    # BUILD GLOBAL ENCODERS
    # ==========================================================
    def buildEncoders(self):
        from sklearn.preprocessing import LabelEncoder
        encs = {}

        for col in self.encoded:
            enc = LabelEncoder()
            values = self.full_df[col].astype(str).unique().tolist()
            enc.fit(values)
            encs[col] = enc

        return encs

    # ==========================================================
    # ENCODING
    # ==========================================================
    def encodeCategoricals(self, df):
        df = df.copy()

        for col, enc in self.encoders.items():
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in enc.classes_ else enc.classes_[0]
            )

            df[col] = enc.transform(df[col])

        return df

    # ==========================================================
    # DECODING
    # ==========================================================
    def decodeCategoricals(self, df):
        for col, enc in self.encoders.items():
            enc = self.encoders[col]

            df[col] = df[col].astype(str).apply(
                lambda x: x if x in enc.classes_ else enc.classes_[0]
            )
            df[col] = enc.transform(df[col].astype(str))
        return df

    # ==========================================================
    # MERGE BASE
    # ==========================================================
    def merge(self, coaches_df, teams_df):
        coaches = coaches_df.copy()
        teams = teams_df.copy()

        coaches = coaches[coaches["stint"].isin([0, 1])].copy()

        protected = {"coachID", "tmID", "year", "lgID", "stint"}
        common_cols = (set(coaches.columns).intersection(set(teams.columns))) - protected

        if common_cols:
            coaches = coaches.rename(columns={c: f"coach_{c}" for c in common_cols})
            teams   = teams.rename(columns={c: f"team_{c}"  for c in common_cols})

        df = coaches.merge(
            teams,
            on=["tmID", "year"],
            how="left"
        )

        return df

    # ==========================================================
    # FEATURE ENGINEERING
    # ==========================================================

    def computeCoachPrevWin(self, df):
        hist = self.coaches_df[["coachID", "year", "won", "lost"]].copy()

        hist["win_pct"] = hist["won"] / (hist["won"] + hist["lost"])

        hist_prev = hist[["coachID", "year", "win_pct"]].copy()
        hist_prev["year"] = hist_prev["year"] + 1
        hist_prev = hist_prev.rename(columns={"win_pct": "prev_coach_win_pct"})

        df2 = df.merge(
            hist_prev,
            on=["coachID", "year"],
            how="left"
        )

        df2["prev_coach_win_pct"] = df2["prev_coach_win_pct"].fillna(0.0)

        return df2

    def computePrevCoachMadePlayoffs(self, df):
        hist = self.coaches_df[["coachID", "year", "post_wins", "post_losses"]].copy()

        hist["made_playoffs"] = ((hist["post_wins"] + hist["post_losses"]) > 0).astype(int)

        hist_prev = hist[["coachID", "year", "made_playoffs"]].copy()
        hist_prev["year"] = hist_prev["year"] + 1
        hist_prev = hist_prev.rename(columns={"made_playoffs": "prev_coach_made_playoffs"})

        df2 = df.merge(
            hist_prev,
            on=["coachID", "year"],
            how="left"
        )

        df2["prev_coach_made_playoffs"] = df2["prev_coach_made_playoffs"].fillna(0).astype(int)

        return df2

    def computeCoachTenure(self, df):
        hist = self.coaches_df[["coachID", "tmID", "year"]].copy()
        hist = hist.sort_values(["coachID", "tmID", "year"])

        hist["coach_tenure"] = 1

        for idx in range(1, len(hist)):
            prev = hist.iloc[idx - 1]
            curr = hist.iloc[idx]

            if (
                curr["coachID"] == prev["coachID"] and
                curr["tmID"] == prev["tmID"] and
                curr["year"] == prev["year"] + 1
            ):
                hist.loc[hist.index[idx], "coach_tenure"] = prev["coach_tenure"] + 1

        df2 = df.merge(
            hist[["coachID", "tmID", "year", "coach_tenure"]],
            on=["coachID", "tmID", "year"],
            how="left"
        )

        df2["coach_tenure"] = df2["coach_tenure"].fillna(1).astype(int)

        return df2

    # ==========================================================
    # MODEL TRAINING
    # ==========================================================
    def generateModel(self, x, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        num_positive = sum(y == 1)
        num_classes = len(np.unique(y))
        if num_classes < 2 or num_positive < 2:
            model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
            model.fit(x, y)
            return model

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

        base = RandomForestClassifier(random_state=42)
        cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        grid = GridSearchCV(
            base,
            param_grid,
            scoring="roc_auc",
            cv=cv_strategy,
            n_jobs=-1,
            verbose=2
        )

        grid.fit(x, y)
        return grid.best_estimator_

    # ==========================================================
    # WALK-FORWARD
    # ==========================================================
    def walkForward(self):
        years = sorted(self.coaches_df["year"].unique())
        results, all_predictions = [], []
        full_df = self.full_df

        for i in range(1, len(years)):
            train_years = years[:i]
            test_year = years[i]

            # split train/test
            train_df = full_df[full_df["year"].isin(train_years)].copy()
            test_df  = full_df[full_df["year"] == test_year].copy()

            # train preprocessing and model training
            X_train, y_train, df_full = self.preprocessTraining(train_df)
            self.model = self.generateModel(X_train, y_train)

            # test preprocessing and predictions
            X_test, df_test_proc = self.preprocessInput(test_df)
            preds = self.model.predict(X_test)

            # debug features input
            self.writeDebugFeatures(X_test, test_year)

            # prediction probabilities
            try:
                probs_full = self.model.predict_proba(X_test)
                probs = probs_full[:, 1] if probs_full.shape[1] == 2 else [0] * len(preds)
            except:
                probs = [0] * len(preds)

            # compute metrics
            results.append(self.computeMetrics(df_test_proc, preds, probs, test_year))

            # decode team IDs
            decoded_teams = self.encoders["tmID"].inverse_transform(
                df_test_proc["tmID"].values
            )

            # store year predictions
            df_out = pd.DataFrame({
                "tmID": decoded_teams,
                "year": test_year,
                "stint": df_test_proc["stint"].values,
                "stint_pred": preds,
                "prob_change": (probs * 100).round().astype(int)
            })

            all_predictions.append(df_out)

        # print metrics
        self.printResults(results)

        # save predictions
        os.makedirs("coach_predictions", exist_ok=True)
        full_df_out = pd.concat(all_predictions, ignore_index=True)
        full_df_out.to_csv("coach_predictions/rfCoach_predictions.csv", index=False)


    # ==========================================================
    # AUXILIARY FUCTIONS
    # ==========================================================
    def writeDebugFeatures(self, df_test_proc, test_year):
        df_debug = df_test_proc.copy()

        for col, enc in self.encoders.items():
            df_debug[col] = self.encoders[col].inverse_transform(
                df_debug[col].astype(int).values
            )

        os.makedirs("coach_debug", exist_ok=True)
        with open(f"coach_debug/debug_features_year_{test_year}.txt", "w") as f:
            f.write(df_debug.to_string(index=False))

    def printResults(self, results):
        print(f"")
        print(f"{'YEAR':<5} | {'PRECISION %':<12} | {'PRECISION':<12} | {'RECALL %':<12} | {'RECALL':<12}")

        for y, tp, fp, fn in results:
            tent = tp + fp
            total_positivos = tp + fn

            precision_pct = f"{tp/tent:.0%}" if tent > 0 else "--"
            recall_pct = f"{tp/total_positivos:.0%}" if total_positivos > 0 else "--"

            precision_str = f"{tp}/{tent}"
            recall_str = f"{tp}/{total_positivos}"

            print(
                f"{y:<5} | "
                f"{precision_pct:<12} | "
                f"{precision_str:<12} | "
                f"{recall_pct:<12} | "
                f"{recall_str:<12}"
            )
        print(f"")

    def computeMetrics(self, df_test_proc, preds, probs, test_year):
        from sklearn.metrics import confusion_matrix

        real = df_test_proc["stint"].astype(int).values

        tn, fp, fn, tp = confusion_matrix(real, preds, labels=[0, 1]).ravel()

        return [test_year, tp, fp, fn]



def main():
    coaches = pd.read_csv("../database/final/coaches.csv")
    teams = pd.read_csv("../database/final/teams.csv")

    model = CoachesModel(coaches, teams)
    model.walkForward()

if __name__ == "__main__":
    main()
