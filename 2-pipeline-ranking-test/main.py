import os
import pandas as pd
from threading import Thread
from playerStats import PlayerStats
from teamRanks import TeamRanks
from teamStats import TeamStats

def player_stats(input_df, test_years):
    # if not os.path.exists("outputs/predicted_players_stats.csv"):
        training_df = pd.read_csv("../database/final/players_teams.csv")
        training_df = training_df[~(training_df["year"].isin(test_years))]
        training_df = training_df.reset_index(drop=True)
        ps_model = PlayerStats(training_df)
        print("PlayerStats model trained!")
    
        ps_input = ps_model.preprocessInput(input_df)
        print("PlayerStats results generated!")
        return ps_model.generateResult(ps_input)
    # else:
    #     print("[CACHED] PlayerStats")
    #     return pd.read_csv("outputs/predicted_players_stats.csv")
        
def team_stats(input_df, test_years):
    # if not os.path.exists("outputs/predicted_team_stats.csv"):
        training_df = pd.read_csv("../database/final/teams.csv")
        training_df = training_df[~training_df["year"].isin(test_years)]
        ts_model = TeamStats(training_df)
        print("TeamStats model trained!")
    
        ts_input = ts_model.preprocessInput(input_df)
        ts_input = TeamStats.filterFeatures(ts_input)
        print("TeamStats results generated!")
        return ts_model.generateResults(ts_input)
    # else:
    #     print("[CACHED] TeamStats")
    #     return pd.read_csv("outputs/predicted_team_stats.csv")
        
def team_ranks(input_df, test_years):
    training_df = pd.read_csv("../database/final/teams.csv")
    training_df = training_df[~training_df["year"].isin(test_years)]
    tr_model = TeamRanks(training_df)
    print("TeamRanks model trained!")

    tr_input = tr_model.preprocessInput(input_df)
    print("TeamStats results generated!")
    return tr_model.generateResults(tr_input)
    
def generate_results(input_df, test_years):
    ps_output = player_stats(input_df, test_years)      # Step 1: Generate player stats from teammates
    ts_output = team_stats(ps_output, test_years)       # Step 2: Generate team stats from player stats
    tr_output = team_ranks(ts_output, test_years)       # Step 3: Generate team ranks from team stats
    return tr_output

def test_one_year(input_df, year, thread, results):
    print("Year: ", year)
    
    # Simulate the input file
    test_years = [year]
    this_input_df = input_df[["playerID", "year", "tmID", "stint"]]
    this_input_df = this_input_df[this_input_df["year"].isin(test_years)]
    this_input_df = this_input_df.reset_index(drop=True)
    print("Input loaded!")
    
    results[thread] = generate_results(this_input_df, test_years)

def test(input_df):
    years = sorted(input_df["year"].unique())
    threads = [None] * len(years)
    results = [None] * len(years)
    
    for i in range(len(threads)):
        threads[i] = Thread(target=test_one_year, args=(input_df, years[i], i, results))
        threads[i].start()
        
    for i in range(len(threads)):
        threads[i].join()

    print(results)

def main():
    input_df = pd.read_csv("../database/final/players_teams.csv")
    test(input_df)

if __name__ == "__main__":
    main()
