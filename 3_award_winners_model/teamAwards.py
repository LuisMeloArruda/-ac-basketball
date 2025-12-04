import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class TeamAwards:
    def __init__(self, data_path="../database/final/"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}

        # Award categories - these are special decade awards
        self.team_awards = [
            "WNBA All-Decade Team",
            "WNBA All Decade Team Honorable Mention",
        ]

    def load_data(self):
        # print("Loading data...")
        self.players_teams = pd.read_csv(f"{self.data_path}players_teams.csv")
        self.players = pd.read_csv(f"{self.data_path}players.csv")
        self.teams = pd.read_csv(f"{self.data_path}teams.csv")
        self.awards = pd.read_csv(f"{self.data_path}awards_players.csv")

        self.awards = self.awards[self.awards["award"].isin(self.team_awards)]

        # print(f"Loaded {len(self.players_teams)} player-season records")
        # print(f"Loaded {len(self.awards)} award records")

    def engineer_decade_features(self, decade_year):
        """
        - Career totals up to the decade year
        - Career averages
        - Number of seasons played
        - Team success (playoffs, championships)
        - Individual awards won
        - Peak season performance
        """
        # Get all seasons up to (but not including) the decade year
        df = self.players_teams[self.players_teams["year"] < decade_year].copy()

        if len(df) == 0:
            return pd.DataFrame()

        # Merge with player bio data
        df = df.merge(
            self.players[["bioID", "pos", "height", "weight", "college"]],
            left_on="playerID",
            right_on="bioID",
            how="left",
        )

        # Calculate per-game stats for each season
        df["ppg"] = df["points"] / df["GP"].replace(0, 1)
        df["rpg"] = (df["oRebounds"] + df["dRebounds"]) / df["GP"].replace(0, 1)
        df["apg"] = df["assists"] / df["GP"].replace(0, 1)
        df["spg"] = df["steals"] / df["GP"].replace(0, 1)
        df["bpg"] = df["blocks"] / df["GP"].replace(0, 1)
        df["fg_pct"] = df["fgMade"] / df["fgAttempted"].replace(0, 1)

        # Aggregate career statistics by player
        career_stats = (
            df.groupby("playerID")
            .agg(
                {
                    # Career totals
                    "points": "sum",
                    "oRebounds": "sum",
                    "dRebounds": "sum",
                    "assists": "sum",
                    "steals": "sum",
                    "blocks": "sum",
                    "GP": "sum",
                    "GS": "sum",
                    "minutes": "sum",
                    "fgMade": "sum",
                    "fgAttempted": "sum",
                    "threeMade": "sum",
                    # Peak performance (max per-game averages)
                    "ppg": "max",
                    "rpg": "max",
                    "apg": "max",
                    "spg": "max",
                    "bpg": "max",
                    "fg_pct": "max",
                    # Seasons played
                    "year": "nunique",
                }
            )
            .reset_index()
        )

        # Rename columns
        career_stats.rename(
            columns={
                "points": "career_points",
                "oRebounds": "career_oreb",
                "dRebounds": "career_dreb",
                "assists": "career_assists",
                "steals": "career_steals",
                "blocks": "career_blocks",
                "GP": "career_games",
                "GS": "career_starts",
                "minutes": "career_minutes",
                "fgMade": "career_fgm",
                "fgAttempted": "career_fga",
                "threeMade": "career_3pm",
                "ppg": "peak_ppg",
                "rpg": "peak_rpg",
                "apg": "peak_apg",
                "spg": "peak_spg",
                "bpg": "peak_bpg",
                "fg_pct": "peak_fg_pct",
                "year": "seasons_played",
            },
            inplace=True,
        )

        # Calculate career averages
        career_stats["career_ppg"] = (
            career_stats["career_points"] / career_stats["career_games"]
        )
        career_stats["career_rpg"] = (
            career_stats["career_oreb"] + career_stats["career_dreb"]
        ) / career_stats["career_games"]
        career_stats["career_apg"] = (
            career_stats["career_assists"] / career_stats["career_games"]
        )
        career_stats["career_spg"] = (
            career_stats["career_steals"] / career_stats["career_games"]
        )
        career_stats["career_bpg"] = (
            career_stats["career_blocks"] / career_stats["career_games"]
        )
        career_stats["career_fg_pct"] = (
            career_stats["career_fgm"] / career_stats["career_fga"]
        )
        career_stats["career_mpg"] = (
            career_stats["career_minutes"] / career_stats["career_games"]
        )
        career_stats["start_pct"] = (
            career_stats["career_starts"] / career_stats["career_games"]
        )

        # Team success metrics
        team_success = df.merge(
            self.teams[["year", "tmID", "won", "playoff"]], on=["year", "tmID"]
        )
        team_success["made_playoffs"] = (team_success["playoff"] != "N").astype(int)
        team_success["won_championship"] = (
            team_success["playoff"].str.contains("W", na=False)
        ).astype(int)

        player_team_success = (
            team_success.groupby("playerID")
            .agg(
                {
                    "made_playoffs": "sum",
                    "won_championship": "sum",
                    "won": "mean",  # Average team wins
                }
            )
            .reset_index()
        )
        player_team_success.rename(
            columns={
                "made_playoffs": "playoff_appearances",
                "won_championship": "championships",
                "won": "avg_team_wins",
            },
            inplace=True,
        )

        career_stats = career_stats.merge(
            player_team_success, on="playerID", how="left"
        )

        # Individual awards won (from awards dataset)
        individual_awards = self.awards[
            (self.awards["year"] < decade_year)
            & (~self.awards["award"].isin(self.team_awards))
        ]
        award_counts = (
            individual_awards.groupby("playerID").size().reset_index(name="awards_won")
        )
        career_stats = career_stats.merge(award_counts, on="playerID", how="left")
        career_stats["awards_won"] = career_stats["awards_won"].fillna(0)

        # Add the decade year
        career_stats["year"] = decade_year

        # Fill NaN values
        numeric_cols = career_stats.select_dtypes(include=[np.number]).columns
        career_stats[numeric_cols] = career_stats[numeric_cols].fillna(0)

        # Replace infinite values
        career_stats = career_stats.replace([np.inf, -np.inf], 0)

        return career_stats

    def create_training_data(self, award_name):
        # Get years when this award was given
        award_years = self.awards[self.awards["award"] == award_name]["year"].unique()

        if len(award_years) == 0:
            return None, None, None, None

        all_data = []

        for year in award_years:
            df = self.engineer_decade_features(year)

            if len(df) == 0:
                continue

            # Create labels for this decade
            award_winners = self.awards[
                (self.awards["award"] == award_name) & (self.awards["year"] == year)
            ]
            df["won_award"] = df["playerID"].isin(award_winners["playerID"]).astype(int)

            all_data.append(df)

        if len(all_data) == 0:
            return None, None, None, None

        df = pd.concat(all_data, ignore_index=True)

        # Feature selection
        exclude_cols = ["playerID", "year", "won_award", "bioID"]

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and df[col].dtype in [np.int64, np.float64]
        ]

        X = df[feature_cols]
        y = df["won_award"]

        return X, y, df, feature_cols

    def train_award_model(self, award_name):
        # print(f"\nTraining model for: {award_name}")

        result = self.create_training_data(award_name)
        if result[0] is None:
            print(f"  No training data available for {award_name}. Skipping.")
            return None, None

        X, y, df, feature_cols = result

        if len(y[y == 1]) < 2:
            print(f"  Insufficient positive samples for {award_name}. Skipping.")
            return None, None

        # print(f"  Total samples: {len(X)}, Positive samples: {y.sum()}")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model - note: these awards have multiple winners
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
        # print("Training Team Award Prediction Models")
        # print("=" * 60)

        for award in self.team_awards:
            model, scaler = self.train_award_model(award)
            if model is not None:
                self.models[award] = {
                    "model": model,
                    "scaler": scaler,
                }

        # print(f"\n{'=' * 60}")
        # print(f"Successfully trained {len(self.models)} award models")
        # print("=" * 60)

    def predict_award_winners(self, decade_year, top_n=10):
        predictions = []

        for award_name, model_info in self.models.items():
            model = model_info["model"]
            scaler = model_info["scaler"]

            df = self.engineer_decade_features(decade_year)

            if len(df) == 0:
                continue

            # Prepare features
            result = self.create_training_data(award_name)
            if result[0] is None:
                continue

            _, _, _, feature_cols = result

            # Filter to only the features in the model
            X_pred = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
            X_pred_scaled = scaler.transform(X_pred)

            # Predict probabilities
            probabilities = model.predict_proba(X_pred_scaled)[:, 1]

            # Get top N candidates (these awards have multiple winners)
            n_winners = 5 if "Honorable" in award_name else 10
            top_indices = np.argsort(probabilities)[-n_winners:][::-1]

            print(f"\n{award_name}:")
            for idx in top_indices:
                winner_id = df.iloc[idx]["playerID"]
                confidence = probabilities[idx]

                predictions.append(
                    {
                        "award": award_name,
                        "year": decade_year,
                        "playerID": winner_id,
                        "confidence": confidence,
                    }
                )

                print(f"  {winner_id} (confidence: {confidence:.3f})")

        return pd.DataFrame(predictions)

    def evaluate_model(self, test_years=None):
        if test_years is None:
            test_years = sorted(self.awards["year"].unique())

        # print(f"\nEvaluating on years: {test_years}")
        # print("=" * 60)

        for year in test_years:
            # print(f"\nDecade Year {year}:")
            predictions = self.predict_award_winners(year)

            # Compare with actual winners
            actual_awards = self.awards[self.awards["year"] == year]

            for award in self.team_awards:
                award_preds = predictions[predictions["award"] == award]
                award_actuals = actual_awards[actual_awards["award"] == award]

                if len(award_actuals) > 0:
                    pred_ids = set(award_preds["playerID"].tolist())
                    actual_ids = set(award_actuals["playerID"].tolist())

                    correct = len(pred_ids & actual_ids)
                    total_actual = len(actual_ids)

                    print(f"\n  {award}:")
                    print(f"    Predicted correctly: {correct}/{total_actual}")
                    print(f"    Predicted: {sorted(pred_ids)}")
                    print(f"    Actual: {sorted(actual_ids)}")


def main():
    # WARN: There is only 1 data point to use, so it is impossible to test this model
    predictor = TeamAwards(data_path="../database/final/")
    predictor.load_data()

    test_year = 11

    predictor.players_teams = predictor.players_teams[
        predictor.players_teams["year"] < test_year
    ]
    predictor.teams = predictor.teams[predictor.teams["year"] < test_year]
    predictor.awards = predictor.awards[predictor.awards["year"] < test_year]

    predictor.train_all_models()

    # Reload full data for evaluation
    predictor.load_data()

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST DATA")
    print("=" * 60)
    predictor.evaluate_model(test_years=[test_year])


if __name__ == "__main__":
    main()
