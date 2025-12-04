import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class PlayerAwards:
    def __init__(self, data_path="../database/final/"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}

        # Award categories
        self.player_awards = [
            "All-Star Game Most Valuable Player",
            "Defensive Player of the Year",
            "Kim Perrot Sportsmanship Award",
            "Most Improved Player",
            "Most Valuable Player",
            "Rookie of the Year",
            "Sixth Woman of the Year",
            "WNBA Finals Most Valuable Player",
        ]

    def load_data(self):
        # print("Loading data...")
        self.players_teams = pd.read_csv(f"{self.data_path}players_teams.csv")
        self.players = pd.read_csv(f"{self.data_path}players.csv")
        self.teams = pd.read_csv(f"{self.data_path}teams.csv")
        self.awards = pd.read_csv(f"{self.data_path}awards_players.csv")

        self.awards = self.awards[self.awards["award"].isin(self.player_awards)]

        # print(f"Loaded {len(self.players_teams)} player-season records")
        # print(f"Loaded {len(self.awards)} award records")

    def engineer_player_features(self, year=None):
        """
        - Current season stats (normalized per game)
        - Team performance
        - Previous year stats and performance delta
        - League rankings
        - Career stats
        """
        df = self.players_teams.copy()

        if year is not None:
            df = df[df["year"] == year]

        # Merge with player bio data
        df = df.merge(
            self.players[["bioID", "pos", "height", "weight", "college"]],
            left_on="playerID",
            right_on="bioID",
            how="left",
        )

        # Per-game statistics
        df["ppg"] = df["points"] / df["GP"].replace(0, 1)  # Points per game
        df["rpg"] = (df["oRebounds"] + df["dRebounds"]) / df["GP"].replace(
            0, 1
        )  # Rebounds per game
        df["apg"] = df["assists"] / df["GP"].replace(0, 1)  # Assists per game
        df["spg"] = df["steals"] / df["GP"].replace(0, 1)  # Steals per game
        df["bpg"] = df["blocks"] / df["GP"].replace(0, 1)  # Blocks per game
        df["topg"] = df["turnovers"] / df["GP"].replace(0, 1)  # Turnovers per game
        df["mpg"] = df["minutes"] / df["GP"].replace(0, 1)  # Minutes played per game

        # Shooting percentages
        df["fg_pct"] = df["fgMade"] / df["fgAttempted"].replace(
            0, 1
        )  # % of Field goald made
        df["ft_pct"] = df["ftMade"] / df["ftAttempted"].replace(
            0, 1
        )  # % of Free throws made
        df["three_per_game"] = df["threeMade"] / df["GP"].replace(
            0, 1
        )  # Nº of "three" type of points made per game

        # Efficiency metrics
        df["true_shooting"] = df["points"] / (
            2 * (df["fgAttempted"] + 0.44 * df["ftAttempted"])
        ).replace(0, 1)
        df["assist_to_turnover"] = df["assists"] / df["turnovers"].replace(0, 1)

        # Games started ratio (indicator of importance)
        df["gs_ratio"] = df["GS"] / df["GP"].replace(0, 1)

        # Merge with team performance
        team_stats = self.teams[
            ["year", "tmID", "won", "lost", "playoff", "o_pts", "d_pts"]
        ].copy()

        team_stats["win_pct"] = team_stats["won"] / (
            team_stats["won"] + team_stats["lost"]
        )  # % of victories

        team_stats["playoff_binary"] = (team_stats["playoff"] != "N").astype(int)

        df = df.merge(team_stats, on=["year", "tmID"], how="left")

        # League rankings for each year (percentile-based)
        for stat in ["ppg", "rpg", "apg", "spg", "bpg", "fg_pct", "true_shooting"]:
            if stat in df.columns:
                df[f"{stat}_rank"] = df.groupby("year")[stat].rank(pct=True)

        # Previous year stats
        df_prev = df.copy()
        df_prev["year"] = df_prev["year"] + 1
        prev_cols = ["playerID", "year", "ppg", "rpg", "apg", "GP"]
        df_prev = df_prev[prev_cols].rename(
            columns={
                "ppg": "prev_ppg",
                "rpg": "prev_rpg",
                "apg": "prev_apg",
                "GP": "prev_GP",
            }
        )

        df = df.merge(df_prev, on=["playerID", "year"], how="left")

        # Improvement metrics
        df["ppg_improvement"] = df["ppg"] - df["prev_ppg"].fillna(df["ppg"])
        df["rpg_improvement"] = df["rpg"] - df["prev_rpg"].fillna(df["rpg"])
        df["apg_improvement"] = df["apg"] - df["prev_apg"].fillna(df["apg"])

        # Rookie indicator (no previous year GP)
        df["is_rookie"] = df["prev_GP"].isna().astype(int)

        # Career years (count of previous seasons)
        career_years = df.groupby("playerID")["year"].rank(method="dense") - 1
        df["career_years"] = career_years

        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def create_training_data(self, award_name):
        df = self.engineer_player_features()
        id_col = "playerID"

        # Create labels
        award_winners = self.awards[self.awards["award"] == award_name]
        df["won_award"] = df.apply(
            lambda row: 1
            if (
                (award_winners["year"] == row["year"])
                & (award_winners[id_col] == row[id_col])
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
            "bioID",
            "pos",
            "college",
            "stint",
            "playoff",
            "lgID",
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

        # Use all data for training since we want to predict the next year
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
        # print("Training Award Prediction Models")
        # print("=" * 60)

        for award in self.player_awards:
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

            df = self.engineer_player_features(year=year)
            id_col = "playerID"

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
                    id_col: winner_id,
                    "confidence": confidence,
                }
            )

            print(f"{award_name}: {winner_id} (confidence: {confidence:.3f})")

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
            # print(f"\nYear {year}:")
            predictions = self.predict_award_winners(year)

            # Compare with actual winners
            actual_awards = self.awards[self.awards["year"] == year]

            for _, pred in predictions.iterrows():
                award = pred["award"]
                actual = actual_awards[actual_awards["award"] == award]

                if len(actual) > 0:
                    pred_id = pred.get("playerID", pred.get("coachID"))
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
    predictor = PlayerAwards(data_path="../database/final/")
    predictor.load_data()

    test_year = 10

    predictor.players_teams = predictor.players_teams[
        predictor.players_teams["year"] < test_year
    ]
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
