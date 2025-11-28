import os
from threading import Thread

import pandas as pd
from playerStats import PlayerStats
from sklearn.metrics import classification_report
from teamRanks import TeamRanks
from teamStats import TeamStats


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


def team_stats(input_df, test_years):
    # if not os.path.exists("outputs/predicted_team_stats.csv"):
        training_df = pd.read_csv("../database/final/teams.csv")
        training_df = training_df[~training_df["year"].isin(test_years)]
        ts_model = TeamStats(training_df)
        print(test_years, "TeamStats model trained!")
    
        ts_input = ts_model.preprocessInput(input_df)
        ts_input = TeamStats.filterFeatures(ts_input)
        ts_output = ts_model.generateResults(ts_input)
        print(test_years, "TeamStats results generated!")
        return ts_output
    # else:
    #     print("[CACHED] TeamStats")
    #     return pd.read_csv("outputs/predicted_team_stats.csv")


def team_ranks(input_df, test_years):
    training_df = pd.read_csv("../database/final/teams.csv")
    training_df = training_df[~training_df["year"].isin(test_years)]
    tr_model = TeamRanks(training_df)
    print(test_years, "TeamRanks model trained!")

    tr_input = tr_model.preprocessInput(input_df)
    tr_output = tr_model.generateResults(tr_input)
    print(test_years, "TeamStats results generated!")
    return tr_output


def generate_results(player_df, team_df, test_years):
    # Step 1: Generate player stats from teammates
    ps_output = player_stats(player_df, test_years)
    
    # Step 2: Generate team stats from player stats
    ts_output = team_stats(ps_output, test_years)
    
    # Step 3: Generate team ranks from team stats
    tr_input = pd.merge(ts_output, team_df, on=["tmID", "year"])
    tr_output = team_ranks(tr_input, test_years)
    return tr_output


def test_one_year(player_df, team_df, year, thread, results):
    # Simulate the input file
    test_years = [year]
    
    this_player_df = player_df[["playerID", "year", "tmID", "stint"]]
    this_player_df = this_player_df[this_player_df["year"].isin(test_years)]
    this_player_df = this_player_df.reset_index(drop=True)
    
    this_team_df = team_df[["year", "tmID", "confID"]]
    this_team_df = this_team_df[this_team_df["year"].isin(test_years)]
    this_team_df = this_team_df.reset_index(drop=True)

    print(test_years, "Input loaded!")

    results[thread] = generate_results(this_player_df, this_team_df, test_years)


def test(player_df, team_df):
    years = sorted(player_df["year"].unique())
    threads = [None] * len(years)
    results = [None] * len(years)

    for i in range(len(threads)):
        threads[i] = Thread(target=test_one_year, args=(player_df, team_df, years[i], i, results))
        threads[i].start()

    for i in range(len(threads)):
        threads[i].join()

    predictions_df = pd.concat(results, ignore_index=True)

    predictions = predictions_df["rank"]
    teams_df = pd.read_csv("../database/final/teams.csv")
    actual = []
    for _, prediction_row in predictions_df.iterrows():
        team = prediction_row["tmID"]
        year = prediction_row["year"]
        rank_answer = teams_df[(teams_df["tmID"] == team) & (teams_df["year"] == year)][
            "rank"
        ].values[0]
        actual.append(rank_answer)

    print(classification_report(actual, predictions))


def main():
    player_df = pd.read_csv("../database/final/players_teams.csv")
    team_df = pd.read_csv("../database/final/teams.csv")
    test(player_df, team_df)


if __name__ == "__main__":
    main()
