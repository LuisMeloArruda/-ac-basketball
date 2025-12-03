import os
from threading import Thread

from sklearn.preprocessing import LabelEncoder
from playerAwards import PlayerAwards
from playerStats import PlayerStats
import pandas as pd
from sklearn.metrics import classification_report


def player_stats(input_df, test_years):
    # if not os.path.exists("outputs/predicted_players_stats.csv"):
        training_df = pd.read_csv("../database/final/players_teams.csv")
        training_df = training_df[~(training_df["year"].isin(test_years))]
        training_df = training_df.reset_index(drop=True)
        encoders_df = pd.concat([training_df, input_df], ignore_index=True)
        encoders = PlayerStats.generateEncoders(encoders_df)
        ps_model = PlayerStats(training_df, encoders)
        print(test_years, "PlayerStats model trained!")
    
        ps_input = ps_model.preprocessInput(input_df)
        ps_output = ps_model.generateResult(ps_input)
        print(test_years, "PlayerStats results generated!")
        return ps_output
    # else:
    #     print("[CACHED] PlayerStats")
    #     return pd.read_csv("outputs/predicted_players_stats.csv")


def player_awards(input_df, test_years):
    players_df = pd.read_csv("../database/final/players_teams.csv")
    awards_df = pd.read_csv("../database/final/awards_players.csv")
    awards_df = awards_df[awards_df["award"].isin(PlayerAwards.validAwards())]

    player_test_mask = players_df["year"].isin(test_years)
    awards_test_mask = awards_df["year"].isin(test_years)

    training_players_df = players_df[~player_test_mask].reset_index(drop=True)
    training_awards_df = awards_df[~awards_test_mask].reset_index(drop=True)
    playerIDs = list(players_df["playerID"].values) + list(input_df["playerID"].values)
    player_encoder = LabelEncoder().fit(playerIDs)
    model = PlayerAwards(training_players_df, training_awards_df, player_encoder)
    print(test_years, "PlayerAwards model trained!")

    test_df = model.preprocessTest(input_df)
    pa_output = model.generateResults(test_df)
    print(test_years, "PlayerAwards results generated!")
    return pa_output
    
def team_awards():
    todo()
    
def coach_awards():
    todo()


def generate_results(player_df, test_years):
    player_stats_df = player_stats(player_df, test_years)
    player_awards_df = player_awards(player_stats_df, test_years)
    # team_awards = team_awards(_, test_years) # TODO: Argument
    # coach_awards = coach_awards(_, test_years) # TODO: Argument
    
    # return pd.concat([player_awards, team_awards, coach_awards], ignore_index=True)
    return player_awards_df


def test_one_year(player_df, year, thread, results):
    # Simulate the input file
    test_years = [year]
    
    this_player_df = player_df[["playerID", "year", "tmID", "stint"]]
    this_player_df = this_player_df[this_player_df["year"].isin(test_years)]
    this_player_df = this_player_df.reset_index(drop=True)

    print(test_years, "Input loaded!")

    results[thread] = generate_results(this_player_df, test_years)


def test(player_df):
    years = sorted(player_df["year"].unique())
    threads = [None] * len(years)
    results = [None] * len(years)

    for i in range(len(threads)):
        threads[i] = Thread(target=test_one_year, args=(player_df, years[i], i, results))
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()

    predictions_df = pd.concat(results, ignore_index=True)

    awards_df = pd.read_csv("../database/final/awards_players.csv")
    actual = []
    predictions = []
    for _, prediction_row in predictions_df.iterrows():
        award = prediction_row["award"]
        year = prediction_row["year"]
        player_predicted = prediction_row["playerID"]
        player_answer = awards_df[(awards_df["award"] == award) & (awards_df["year"] == year)]["playerID"].values
        if len(player_answer) == 1:
            actual.append(player_answer[0])
            predictions.append(player_predicted)
        
    print(actual)
    print(predictions)
    print(classification_report(actual, predictions))


def main():
    player_df = pd.read_csv("../database/final/players_teams.csv")
    test(player_df)


if __name__ == "__main__":
    main()
