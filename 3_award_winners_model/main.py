import pandas as pd
from coachAwards import CoachAwards
from coachStats import CoachStats
from playerAwards import PlayerAwards
from playerStats import PlayerStats
from sklearn.preprocessing import LabelEncoder
from teamAwards import TeamAwards
from teamStats import TeamStats


def main():
    data_path = "../database/final/"
    test_year = 10
    training_years = list(range(1, test_year))

    # 1. Load data
    players_teams_full = pd.read_csv(f"{data_path}players_teams.csv")
    coaches_full = pd.read_csv(f"{data_path}coaches.csv")
    teams_full = pd.read_csv(f"{data_path}teams.csv")
    players_bio = pd.read_csv(f"{data_path}players.csv")
    awards_full = pd.read_csv(f"{data_path}awards_players.csv")

    # 2. Prepare/Filter training data

    players_teams_train = players_teams_full[
        players_teams_full["year"].isin(training_years)
    ].copy()
    players_teams_train.reset_index(drop=True, inplace=True)
    coaches_train = coaches_full[coaches_full["year"].isin(training_years)].copy()
    coaches_train.reset_index(drop=True, inplace=True)
    teams_train = teams_full[teams_full["year"].isin(training_years)].copy()
    teams_train.reset_index(drop=True, inplace=True)
    awards_train = awards_full[awards_full["year"].isin(training_years)].copy()
    awards_train.reset_index(drop=True, inplace=True)

    # 3. Train Statistics Models
    player_encoders = PlayerStats.generateEncoders(players_teams_full)
    player_stats_model = PlayerStats(players_teams_train, player_encoders)

    team_stats_model = TeamStats(teams_train)

    coach_encoder = LabelEncoder().fit(coaches_full["coachID"])
    team_encoder = LabelEncoder().fit(coaches_full["tmID"])
    coach_stats_model = CoachStats(
        coaches_train, teams_train, coach_encoder, team_encoder
    )

    # 4. Train Award Models
    player_award_model = PlayerAwards(data_path=data_path)
    player_award_model.players_teams = players_teams_train
    player_award_model.players = players_bio
    player_award_model.teams = teams_train
    player_award_model.awards = awards_train
    player_award_model.train_all_models()

    coach_award_model = CoachAwards(data_path=data_path)
    coach_award_model.coaches = coaches_train
    coach_award_model.teams = teams_train
    coach_award_model.awards = awards_train
    coach_award_model.train_all_models()

    team_award_model = TeamAwards(data_path=data_path)
    team_award_model.players_teams = players_teams_train
    team_award_model.players = players_bio
    team_award_model.teams = teams_train
    team_award_model.awards = awards_train
    team_award_model.train_all_models()

    # 5. Prepare Year 10 Input Data
    players_roster_10 = players_teams_full[players_teams_full["year"] == test_year][
        ["playerID", "year", "tmID", "stint"]
    ].copy()
    players_roster_10.reset_index(drop=True, inplace=True)

    coaches_roster_10 = coaches_full[coaches_full["year"] == test_year][
        ["coachID", "year", "tmID", "stint"]
    ].copy()
    coaches_roster_10.reset_index(drop=True, inplace=True)

    teams_roster_10 = teams_full[teams_full["year"] == test_year][
        ["year", "tmID", "franchID", "confID"]
    ].copy()
    teams_roster_10.reset_index(drop=True, inplace=True)

    # 6. Generate Statistics for Year 10
    players_roster_preprocessed = player_stats_model.preprocessInput(
        players_roster_10[["playerID", "year", "tmID", "stint"]]
    )
    player_stats_10 = player_stats_model.generateResult(players_roster_preprocessed)

    team_input = team_stats_model.preprocessInput(player_stats_10)
    team_stats_10 = team_stats_model.generateResults(team_input)

    coaches_roster_preprocessed = coach_stats_model.preprocessInput(
        coaches_roster_10[["coachID", "year", "tmID", "stint"]]
    )
    coach_stats_10 = coach_stats_model.generateResults(coaches_roster_preprocessed)

    # 7. Merge Generated Stats with Historical Data
    players_teams_with_predictions = pd.concat(
        [players_teams_train, player_stats_10], ignore_index=True
    )

    teams_with_predictions = pd.concat([teams_train, team_stats_10], ignore_index=True)

    coaches_with_predictions = pd.concat(
        [coaches_train, coach_stats_10], ignore_index=True
    )

    # 8. Predict Awards for test year
    player_award_model.players_teams = players_teams_with_predictions
    player_award_model.teams = teams_with_predictions
    player_predictions = player_award_model.predict_award_winners(test_year)

    coach_award_model.coaches = coaches_with_predictions
    coach_award_model.teams = teams_with_predictions
    coach_predictions = coach_award_model.predict_award_winners(test_year)

    team_award_model.players_teams = players_teams_with_predictions
    team_award_model.teams = teams_with_predictions
    team_predictions = team_award_model.predict_award_winners(test_year)

    # 9. Combine and Display Results
    all_predictions = pd.concat(
        [player_predictions, coach_predictions, team_predictions], ignore_index=True
    )

    print(f"\n{'Award':<50} {'Winner':<15}")
    print("-" * 65)
    for _, row in all_predictions.iterrows():
        print(f"{row['award']:<50} {row['playerID']:<15}")

    # 10. Evaluation (Compare with Actual Year 10 Awards)
    actual_awards = awards_full[awards_full["year"] == test_year]

    if len(actual_awards) > 0:
        correct = 0
        total = 0

        print(f"\n{'Award':<50} {'Predicted':<15} {'Actual':<15} {'Match'}")
        print("-" * 85)

        for _, pred_row in all_predictions.iterrows():
            award = pred_row["award"]
            predicted_winner = pred_row["playerID"]

            actual = actual_awards[actual_awards["award"] == award]

            if len(actual) > 0:
                actual_winners = actual["playerID"].tolist()

                # For team awards (multiple winners), check if predicted is among actual
                if len(actual) > 1:
                    match = predicted_winner in actual_winners
                    actual_str = f"{len(actual_winners)} winners"
                else:
                    actual_winner = actual_winners[0]
                    match = predicted_winner == actual_winner
                    actual_str = actual_winner

                match_symbol = "✓" if match else "✗"
                print(
                    f"{award:<50} {predicted_winner:<15} {actual_str:<15} {match_symbol}"
                )

                if match:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        print("\n" + "-" * 85)
        print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    else:
        print(f"\nNo actual awards found for year {test_year} - skipping evaluation")

    # 11. Save Results
    output_file = f"predictions_year_{test_year}.csv"
    all_predictions.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")

    player_stats_10.to_csv(f"generated_player_stats_year_{test_year}.csv", index=False)
    team_stats_10.to_csv(f"generated_team_stats_year_{test_year}.csv", index=False)
    coach_stats_10.to_csv(f"generated_coach_stats_year_{test_year}.csv", index=False)
    print(f"Generated statistics saved in working directory")


if __name__ == "__main__":
    main()
