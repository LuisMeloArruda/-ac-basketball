import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class CoachAwards:
    def __init__(self, data_path="../database/final/"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}

        # Award category
        self.coach_awards = ["Coach of the Year"]

    def load_data(self):
        # print("Loading data...")
        self.coaches = pd.read_csv(f"{self.data_path}coaches.csv")
        self.teams = pd.read_csv(f"{self.data_path}teams.csv")
        self.awards = pd.read_csv(f"{self.data_path}awards_players.csv")

        self.awards = self.awards[self.awards["award"].isin(self.coach_awards)]

        # print(f"Loaded {len(self.coaches)} coach-season records")
        # print(f"Loaded {len(self.awards)} award records")

    def engineer_coach_features(self, year=None):
        """
        - Season record (wins, losses, win percentage)
        - Playoff performance
        - Team improvement metrics
        - Conference performance
        - Historical performance
        """
        df = self.coaches.copy()

        if year is not None:
            df = df[df["year"] == year]

        # Win percentage
        df["win_pct"] = df["won"] / (df["won"] + df["lost"]).replace(0, 1)

        # Playoff metrics
        df["post_win_pct"] = df["post_wins"] / (
            df["post_wins"] + df["post_losses"]
        ).replace(0, 1)
        df["post_win_pct"] = df["post_win_pct"].fillna(0)
        df["made_playoffs"] = ((df["post_wins"] + df["post_losses"]) > 0).astype(int)
        df["total_post_games"] = df["post_wins"] + df["post_losses"]

        # Merge with team stats for additional context
        team_stats = self.teams[
            [
                "year",
                "tmID",
                "playoff",
                "confW",
                "confL",
                "won",
                "lost",
                "o_pts",
                "d_pts",
                "homeW",
                "homeL",
                "awayW",
                "awayL",
            ]
        ].copy()

        team_stats["conf_win_pct"] = team_stats["confW"] / (
            team_stats["confW"] + team_stats["confL"]
        ).replace(0, 1)
        team_stats["home_win_pct"] = team_stats["homeW"] / (
            team_stats["homeW"] + team_stats["homeL"]
        ).replace(0, 1)
        team_stats["away_win_pct"] = team_stats["awayW"] / (
            team_stats["awayW"] + team_stats["awayL"]
        ).replace(0, 1)
        team_stats["avg_pts_scored"] = team_stats["o_pts"] / (
            team_stats["won"] + team_stats["lost"]
        )
        team_stats["avg_pts_allowed"] = team_stats["d_pts"] / (
            team_stats["won"] + team_stats["lost"]
        )
        team_stats["point_differential"] = (
            team_stats["avg_pts_scored"] - team_stats["avg_pts_allowed"]
        )

        # Playoff depth (how far they went)
        team_stats["reached_finals"] = (team_stats["playoff"].str.contains("W")).astype(
            int
        )
        team_stats["reached_semis"] = (team_stats["playoff"].str.len() >= 2).astype(int)

        df = df.merge(
            team_stats, on=["year", "tmID"], how="left", suffixes=("", "_team")
        )

        # Previous year stats for the same coach with same team
        df_prev = df.copy()
        df_prev["year"] = df_prev["year"] + 1
        prev_cols = ["coachID", "tmID", "year", "won", "win_pct", "made_playoffs"]
        df_prev = df_prev[prev_cols].rename(
            columns={
                "won": "prev_won",
                "win_pct": "prev_win_pct",
                "made_playoffs": "prev_made_playoffs",
            }
        )

        df = df.merge(df_prev, on=["coachID", "tmID", "year"], how="left")

        # Improvement metrics (key for Coach of the Year)
        df["wins_improvement"] = df["won"] - df["prev_won"].fillna(0)
        df["win_pct_improvement"] = df["win_pct"] - df["prev_win_pct"].fillna(0)
        df["playoff_improvement"] = df["made_playoffs"] - df[
            "prev_made_playoffs"
        ].fillna(0)

        # New coach indicator (first year with team)
        df["is_new_coach"] = df["prev_won"].isna().astype(int)

        # League rankings for each year (percentile-based)
        for stat in ["win_pct", "won", "conf_win_pct", "point_differential"]:
            if stat in df.columns:
                df[f"{stat}_rank"] = df.groupby("year")[stat].rank(pct=True)

        # Career statistics with this team
        df["years_with_team"] = df.groupby(["coachID", "tmID"])["year"].rank(
            method="dense"
        )

        # Overall coaching experience
        df["coaching_years"] = df.groupby("coachID")["year"].rank(method="dense")

        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def create_training_data(self, award_name):
        df = self.engineer_coach_features()
        id_col = "coachID"

        # Create labels
        award_winners = self.awards[self.awards["award"] == award_name]
        df["won_award"] = df.apply(
            lambda row: 1
            if (
                (award_winners["year"] == row["year"])
                & (
                    award_winners["playerID"] == row[id_col]
                )  # coachID stored as playerID in awards
            ).any()
            else 0,
            axis=1,
        )

        # Filter to years where award was given
        valid_years = award_winners["year"].unique()
        df = df[df["year"].isin(valid_years)]

        # Feature selection
        exclude_cols = [
            id_col,
            "year",
            "won_award",
            "tmID",
            "lgID",
            "stint",
            "playoff",
        ]

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]
        ]

        X = df[feature_cols]
        y = df["won_award"]

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)

        return X, y, df, feature_cols

    def train_award_model(self, award_name):
        # print(f"\nTraining model for: {award_name}")

        X, y, df, feature_cols = self.create_training_data(award_name)

        if len(y[y == 1]) < 2:
            print(f"  Insufficient positive samples for {award_name}. Skipping.")
            return None, None

        # print(f"  Total samples: {len(X)}, Positive samples: {y.sum()}")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model with class balancing
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            subsample=0.8,
        )

        model.fit(X_scaled, y)

        # Feature importance
        importances = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(f"{award_name}: Top 5 important features:")
        for idx, row in importances.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        return model, scaler

    def train_all_models(self):
        # print("=" * 60)
        # print("Training Coach Award Prediction Models")
        # print("=" * 60)

        for award in self.coach_awards:
            model, scaler = self.train_award_model(award)
            if model is not None:
                self.models[award] = {
                    "model": model,
                    "scaler": scaler,
                }

        # print(f"\n{'=' * 60}")
        # print(f"Successfully trained {len(self.models)} award models")
        # print("=" * 60)

    def predict_award_winners(self, year):
        predictions = []

        for award_name, model_info in self.models.items():
            model = model_info["model"]
            scaler = model_info["scaler"]

            df = self.engineer_coach_features(year=year)
            id_col = "coachID"

            if len(df) == 0:
                continue

            # Prepare features (same columns as training)
            X, _, _, feature_cols = self.create_training_data(award_name)

            # Filter to only the features in the model
            X_pred = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
            X_pred_scaled = scaler.transform(X_pred)

            # Predict probabilities
            probabilities = model.predict_proba(X_pred_scaled)[:, 1]

            # Get top candidate
            top_idx = np.argmax(probabilities)
            winner_id = df.iloc[top_idx][id_col]
            confidence = probabilities[top_idx]

            predictions.append(
                {
                    "award": award_name,
                    "year": year,
                    "playerID": winner_id,  # Store as playerID to match awards format
                    "confidence": confidence,
                }
            )

            # print(f"{award_name}: {winner_id} (confidence: {confidence:.3f})")

        return pd.DataFrame(predictions)

    def evaluate_model(self, test_years=None):
        if test_years is None:
            all_years = sorted(self.awards["year"].unique())
            test_years = all_years[-2:]  # Last 2 years

        # print(f"\nEvaluating on years: {test_years}")
        # print("=" * 60)

        all_predictions = []
        all_actuals = []

        for year in test_years:
            print(f"\nYear {year}:")
            predictions = self.predict_award_winners(year)

            # Compare with actual winners
            actual_awards = self.awards[self.awards["year"] == year]

            for _, pred in predictions.iterrows():
                award = pred["award"]
                actual = actual_awards[actual_awards["award"] == award]

                if len(actual) > 0:
                    pred_id = pred["playerID"]
                    actual_id = actual.iloc[0]["playerID"]

                    all_predictions.append(pred_id)
                    all_actuals.append(actual_id)

                    match = "✓" if pred_id == actual_id else "✗"
                    print(f"  {match} {award}")
                    print(f"    Predicted: {pred_id}, Actual: {actual_id}")

        if len(all_predictions) > 0:
            accuracy = accuracy_score(all_actuals, all_predictions)
            print(f"\n{'=' * 60}")
            print(f"Overall Accuracy: {accuracy:.2%}")
            print(
                f"Correct predictions: {sum(p == a for p, a in zip(all_predictions, all_actuals))}/{len(all_predictions)}"
            )
            print("=" * 60)


def main():
    predictor = CoachAwards(data_path="../database/final/")
    predictor.load_data()

    test_year = 11

    predictor.coaches = predictor.coaches[predictor.coaches["year"] < test_year]
    predictor.teams = predictor.teams[predictor.teams["year"] < test_year]
    predictor.awards = predictor.awards[predictor.awards["year"] < test_year]

    predictor.train_all_models()
    predictor.load_data()

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST DATA")
    print("=" * 60)
    predictor.evaluate_model(test_years=[test_year])


if __name__ == "__main__":
    main()
