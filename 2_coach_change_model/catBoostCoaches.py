import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier, Pool


class CoachesModel:
    def __init__(self, coaches_df, teams_df, players_df, N_years=3):
        self.coaches_df = coaches_df.copy()
        self.teams_df = teams_df.copy()
        self.players_df = players_df.copy()

        # CatBoost handles categorical features automatically, but they must be explicitly
        self.categorical_cols = ["tmID", "coachID"]

    # ==========================================================
    # FEATURE SELECTION
    # ==========================================================
    def filterFeatures(self, df):
        feature_cols = [
            "tmID","coachID",
            "prev_coach_won","prev_coach_lost","prev_coach_post_wins","prev_coach_post_losses",
            "prev_team_won","prev_team_lost","prev_team_o_pts","prev_team_d_pts","prev_team_playoff",
            "prev_coach_win_pct","prev_coach_made_playoffs",
            "coach_tenure","roster_talent_past_score",
        ]
        return df[feature_cols]

    # ==========================================================
    # TARGET SELECTION
    # ==========================================================
    def filterTarget(self, df):
        return df["stint"]

    # ==========================================================
    # PREPROCESS
    # ==========================================================
    def preprocess(self, df):
        df = self.computeCoachPrevWin(df)
        df = self.computePrevCoachMadePlayoffs(df)
        df = self.computeCoachTenure(df)
        df = self.computeInheritedTalent(df)
        df = self.bumpPastStats(df)
        return df

    # ==========================================================
    # MERGE BASE TABLES
    # ==========================================================
    def merge(self):
        coaches = self.coaches_df.copy()
        teams = self.teams_df.copy()

        coaches = coaches[coaches["stint"].isin([0, 1])].copy()

        protected = {"coachID", "tmID", "year", "stint"}
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
        h = self.coaches_df.copy()
        h["prev_coach_win_pct"] = (h["won"] / (h["won"] + h["lost"])).replace({np.inf: np.nan})

        h = h[["coachID", "year", "prev_coach_win_pct"]].copy()
        h["year"] += 1

        df2 = df.merge(h, on=["coachID", "year"], how="left")
        return df2


    def computePrevCoachMadePlayoffs(self, df):
        h = self.coaches_df.copy()
        h["prev_coach_made_playoffs"] = ((h["post_wins"] + h["post_losses"]) > 0).astype(float)

        h = h[["coachID", "year", "prev_coach_made_playoffs"]]
        h["year"] += 1

        df2 = df.merge(h, on=["coachID", "year"], how="left")
        return df2


    def computeCoachTenure(self, df):
        h = self.coaches_df[["coachID", "tmID", "year"]].copy().sort_values(["coachID", "tmID", "year"])
        h["coach_tenure"] = 1

        for i in range(1, len(h)):
            p = h.iloc[i - 1]
            c = h.iloc[i]
            if c.coachID == p.coachID and c.tmID == p.tmID and c.year == p.year + 1:
                h.loc[h.index[i], "coach_tenure"] = p["coach_tenure"] + 1

        df2 = df.merge(h, on=["coachID", "tmID", "year"], how="left")
        return df2


    def computeInheritedTalent(self, df):
        p = self.players_df.copy()

        stats_cols = [
            "points", "oRebounds", "dRebounds", "assists", "steals", "blocks",
            "turnovers", "fgAttempted", "fgMade", "ftAttempted", "ftMade"
        ]

        for c in stats_cols:
            if c not in p.columns:
                p[c] = np.nan


        p["efficiency"] = (
            p["points"]
            + p["oRebounds"] + p["dRebounds"]
            + p["assists"] + p["steals"] + p["blocks"]
            - (p["fgAttempted"] - p["fgMade"])
            - (p["ftAttempted"] - p["ftMade"])
            - p["turnovers"]
        )

        hist_next = (
            p[["playerID", "tmID", "year", "efficiency"]]
            .assign(year=lambda x: x["year"] + 1)
            .rename(columns={"efficiency": "eff_next"})
        )

        roster = p[["playerID", "tmID", "year"]].copy()
        merged = roster.merge(hist_next, on=["playerID", "tmID", "year"], how="left")

        talent = (
            merged.groupby(["tmID", "year"])["eff_next"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"eff_next": "roster_talent_past_score"})
        )

        df2 = df.merge(talent, on=["tmID", "year"], how="left")
        return df2


    def bumpPastStats(self, df):
        coach_cols = ["won", "lost", "post_wins", "post_losses"]
        team_cols = [
            "won","lost","GP","homeW","homeL","awayW","awayL",
            "o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_oreb","o_dreb","o_reb",
            "o_asts","o_pf","o_stl","o_to","o_blk","o_pts",
            "d_fgm","d_ftm","d_3pm","d_3pa","d_oreb","d_dreb","d_asts",
            "d_pf","d_stl","d_to","d_blk","d_pts","playoff"
        ]

        # coaches
        c_hist = self.coaches_df[["coachID", "year"] + coach_cols].copy()
        c_hist["year"] += 1
        c_hist = c_hist.rename(columns={c: f"prev_coach_{c}" for c in coach_cols})
        df2 = df.merge(c_hist, on=["coachID", "year"], how="left")

        # teams
        t_hist = self.teams_df[["tmID", "year"] + team_cols].copy()
        t_hist["year"] += 1
        t_hist = t_hist.rename(columns={c: f"prev_team_{c}" for c in team_cols})
        df2 = df2.merge(t_hist, on=["tmID", "year"], how="left")

        if "prev_team_playoff" in df2:
            df2["prev_team_playoff"] = df2["prev_team_playoff"].apply(
                lambda x: 1 if isinstance(x, str) and x.upper() == "Y" else (0 if isinstance(x, str) else np.nan)
            )

        return df2

    # ==========================================================
    # CATBOOST MODEL
    # ==========================================================
    def makeModel(self, X, y):
        model = CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            class_weights=[1, 8],
            verbose=False
        )
        model.fit(Pool(X, y, cat_features=self.categorical_cols))
        return model

    # ==========================================================
    # WALK-FORWARD
    # ==========================================================
    def walkForward(self):
        years = sorted(self.coaches_df["year"].unique())
        base = self.merge()

        results, preds = [], []

        for i in range(1, len(years)):

            # split train/test
            train_years = years[:i]
            test_year = years[i]

            train_df = self.preprocess(base[base.year.isin(train_years)])
            test_df  = self.preprocess(base[base.year == test_year])

            X_train = self.filterFeatures(train_df)
            y_train = self.filterTarget(train_df)

            # train model
            model = self.makeModel(X_train, y_train)

            # test model
            X_test = self.filterFeatures(test_df)
            pool = Pool(X_test, cat_features=self.categorical_cols)

            # debug features input
            self.writeDebugFeatures(X_test, test_year)

            # compute metrics
            pred = model.predict(pool).astype(int)
            prob = model.predict_proba(pool)[:, 1]
            real = test_df["stint"].values

            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(real, pred, labels=[0, 1]).ravel()

            results.append([test_year, tp, fp, fn])

            preds.append(pd.DataFrame({
                "tmID": test_df["tmID"],
                "year": test_year,
                "stint": real,
                "stint_pred": pred,
                "prob_change": (prob*100).round().astype(int)
            }))

        # print metrics
        self.printResults(results)

        # save predictions
        os.makedirs("coach_predictions", exist_ok=True)
        pd.concat(preds).to_csv("coach_predictions/catBoostCoach_predictions.csv", index=False)


    # ==========================================================
    # AUXILIARY FUCTIONS
    # ==========================================================
    def writeDebugFeatures(self, df_test_proc, test_year):
        df_debug = df_test_proc.copy()

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


def main():
    c = pd.read_csv("../database/final/coaches.csv")
    t = pd.read_csv("../database/final/teams.csv")
    p = pd.read_csv("../database/final/players_teams.csv")

    CoachesModel(c, t, p).walkForward()


if __name__ == "__main__":
    main()
