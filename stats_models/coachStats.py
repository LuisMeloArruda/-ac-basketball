import math
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder


class CoachStats:
    def __init__(self, training_df, team_df, coach_encoder, team_encoder):
        self.encoders = {
            "coachID": coach_encoder,
            "tmID": team_encoder
        }
        self.team_df = team_df
        training_df = self.preprocessTraining(training_df)
        self.model = self.trainModel(training_df)

    def preprocessTraining(self, training_df):
        df = training_df.copy()

        # Encode categorical variables
        df["coachID_encoded"] = self.encoders["coachID"].transform(df["coachID"].astype(str))
        df["tmID_encoded"] = self.encoders["tmID"].transform(df["tmID"].astype(str))

        # Add historical features
        df = self._add_historical_features(df)

        return df

    def _add_historical_features(self, df):
        # Sort by coach and year
        df = df.sort_values(["coachID", "year"])

        # Coach career stats (cumulative)
        df["career_games"] = df.groupby("coachID").cumcount() + 1
        df["career_wins"] = df.groupby("coachID")["won"].cumsum()
        df["career_losses"] = df.groupby("coachID")["lost"].cumsum()
        df["career_win_pct"] = df["career_wins"] / (
            df["career_wins"] + df["career_losses"]
        )

        # Previous year performance with this team
        df["prev_won"] = df.groupby(["coachID", "tmID"])["won"].shift(1)
        df["prev_lost"] = df.groupby(["coachID", "tmID"])["lost"].shift(1)

        # Coach's overall previous year performance (any team)
        df["prev_won_any"] = df.groupby("coachID")["won"].shift(1)

        # Team's previous year performance (with any coach)
        team_prev = self.team_df[["tmID", "year", "won", "lost"]].copy()
        team_prev["year"] = team_prev["year"] + 1
        team_prev = team_prev.rename(
            columns={"won": "team_prev_won", "lost": "team_prev_lost"}
        )
        df = df.merge(team_prev, on=["tmID", "year"], how="left")

        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def preprocessInput(self, input_df):
        df = input_df.copy()

        # Get historical data for feature engineering
        historical_coaches = pd.read_csv("../database/final/coaches.csv")
        historical_coaches = historical_coaches[
            historical_coaches["year"] < df["year"].min()
        ]

        # Encode
        df["coachID_encoded"] = self.encoders["coachID"].transform(
            df["coachID"].astype(str)
        )
        df["tmID_encoded"] = self.encoders["tmID"].transform(df["tmID"].astype(str))

        # Add historical features for each coach
        coach_features = []

        for _, row in df.iterrows():
            coach_id = row["coachID"]
            tm_id = row["tmID"]
            year = row["year"]

            # Get coach history
            coach_hist = historical_coaches[historical_coaches["coachID"] == coach_id]

            if len(coach_hist) > 0:
                career_games = len(coach_hist)
                career_wins = coach_hist["won"].sum()
                career_losses = coach_hist["lost"].sum()
                career_win_pct = (
                    career_wins / (career_wins + career_losses)
                    if (career_wins + career_losses) > 0
                    else 0.5
                )

                # Previous year (any team)
                prev_year_data = coach_hist[coach_hist["year"] == year - 1]
                prev_won_any = (
                    prev_year_data["won"].iloc[0] if len(prev_year_data) > 0 else 0
                )

                # Previous year with this team
                coach_team_hist = coach_hist[coach_hist["tmID"] == tm_id]
                if len(coach_team_hist) > 0:
                    prev_won = coach_team_hist.iloc[-1]["won"]
                    prev_lost = coach_team_hist.iloc[-1]["lost"]
                else:
                    prev_won = 0
                    prev_lost = 0
            else:
                # New coach
                career_games = 0
                career_wins = 0
                career_losses = 0
                career_win_pct = 0.5
                prev_won_any = 0
                prev_won = 0
                prev_lost = 0

            # Team's previous performance
            team_hist = self.team_df[
                (self.team_df["tmID"] == tm_id) & (self.team_df["year"] == year - 1)
            ]
            if len(team_hist) > 0:
                team_prev_won = team_hist.iloc[0]["won"]
                team_prev_lost = team_hist.iloc[0]["lost"]
            else:
                team_prev_won = 17
                team_prev_lost = 17

            features = {
                "coachID": coach_id,
                "year": year,
                "tmID": tm_id,
                "coachID_encoded": row["coachID_encoded"],
                "tmID_encoded": row["tmID_encoded"],
                "career_games": career_games,
                "career_wins": career_wins,
                "career_losses": career_losses,
                "career_win_pct": career_win_pct,
                "prev_won": prev_won,
                "prev_lost": prev_lost,
                "prev_won_any": prev_won_any,
                "team_prev_won": team_prev_won,
                "team_prev_lost": team_prev_lost,
            }

            coach_features.append(features)

        return pd.DataFrame(coach_features)

    @staticmethod
    def filterFeatures(df):
        return df[
            [
                "coachID_encoded",
                "tmID_encoded",
                "year",
                "career_games",
                "career_win_pct",
                "prev_won",
                "prev_lost",
                "prev_won_any",
                "team_prev_won",
                "team_prev_lost",
            ]
        ]

    @staticmethod
    def filterTargets(df):
        return df[
            [
                "won",
                "lost",
                "post_wins",
                "post_losses",
            ]
        ]

    def trainModel(self, training_df):
        X = self.filterFeatures(training_df)
        y = self.filterTargets(training_df)

        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        )
        model.fit(X, y)

        return model

    def generateResults(self, input_df):
        X = self.filterFeatures(input_df)
        predictions = self.model.predict(X)

        pred_df = pd.DataFrame(
            predictions, columns=["won", "lost", "post_wins", "post_losses"]
        )

        # Round to integers
        pred_df = pred_df.round(0).astype(int)

        # Ensure valid values
        pred_df["won"] = pred_df["won"].clip(0, 34)
        pred_df["lost"] = pred_df["lost"].clip(0, 34)
        pred_df["post_wins"] = pred_df["post_wins"].clip(0, 10)
        pred_df["post_losses"] = pred_df["post_losses"].clip(0, 10)

        # Add back identifiers
        pred_df["coachID"] = input_df["coachID"].values
        pred_df["year"] = input_df["year"].values
        pred_df["tmID"] = input_df["tmID"].values

        # Reorder columns
        pred_df = pred_df[
            ["coachID", "year", "tmID", "won", "lost", "post_wins", "post_losses"]
        ]

        return pred_df

    def testModel(self, input_df, result_df):
        pred = self.generateResults(input_df)

        # Merge predicted and real
        merged = pred.merge(
            result_df, on=["coachID", "year", "tmID"], suffixes=("_pred", "_real")
        )

        cols = ["won", "lost", "post_wins", "post_losses"]

        y_true = merged[[c + "_real" for c in cols]].values
        y_pred = merged[[c + "_pred" for c in cols]].values

        print("Mean Absolute Error:", round(mean_absolute_error(y_true, y_pred), 4))
        print("Mean Squared Error:", round(mean_squared_error(y_true, y_pred), 4))
        print("R² Score:", round(r2_score(y_true, y_pred), 4))


def main():
    coaches_df = pd.read_csv("../database/final/coaches.csv")
    teams_df = pd.read_csv("../database/final/teams.csv")

    # Split data for testing
    years = coaches_df["year"].unique().tolist()
    random.shuffle(years)
    test_size = 0.1
    test_years = years[: math.ceil(len(years) * test_size)]

    train_mask = ~coaches_df["year"].isin(test_years)
    training_df = coaches_df[train_mask]
    test_df = coaches_df[~train_mask]

    # Train model
    coach_encoder = LabelEncoder().fit(coaches_df["coachID"])
    team_encoder = LabelEncoder().fit(coaches_df["tmID"])
    model = CoachStats(training_df, teams_df, coach_encoder, team_encoder)

    # Prepare test input (simulate having only roster info)
    test_input = test_df[["coachID", "year", "tmID", "stint"]].copy()
    test_input = model.preprocessInput(test_input)

    # Test
    model.testModel(test_input, test_df)


if __name__ == "__main__":
    main()
