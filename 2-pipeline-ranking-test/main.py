import os
import pandas as pd
from playerStats import PlayerStats
from teamRanks import TeamRanks
from teamStats import TeamStats

def player_stats(input_df, test_years):
    if not os.path.exists("outputs/predicted_players_stats.csv"):
        training_df = pd.read_csv("../database/final/players_teams.csv")
        training_df = training_df[~(training_df["year"].isin(test_years))]
        training_df = training_df.reset_index(drop=True)
        ps_model = PlayerStats(training_df)
        print("PlayerStats model trained!")
    
        ps_input = ps_model.preprocessInput(input_df)
        print("PlayerStats results generated!")
        return ps_model.generateResult(ps_input)
    else:
        print("[CACHED] PlayerStats")
        return pd.read_csv("outputs/predicted_players_stats.csv")
        
def team_stats(input_df, test_years):
    if not os.path.exists("outputs/predicted_team_stats.csv"):
        training_df = pd.read_csv("../database/final/teams.csv")
        training_df = training_df[~training_df["year"].isin(test_years)]
        ts_model = TeamStats(training_df)
        print("TeamStats model trained!")
    
        ts_input = ts_model.preprocessInput(input_df)
        ts_input = TeamStats.filterFeatures(ts_input)
        print("TeamStats results generated!")
        return ts_model.generateResults(ts_input)
    else:
        print("[CACHED] TeamStats")
        return pd.read_csv("outputs/predicted_team_stats.csv")
        
def team_ranks(input_df, test_years):
    training_df = pd.read_csv("../database/final/teams.csv")
    training_df = training_df[~training_df["year"].isin(test_years)]
    tr_model = TeamRanks(training_df)
    print("TeamRanks model trained!")

    (tr_input, _) = TeamRanks.preprocess(input_df)
    tr_input = TeamRanks.filterFeatures(tr_input)
    print("TeamStats results generated!")
    return TeamRanks.generateResults(tr_model.model, tr_input, tr_model.encoders)

def main():
    # Simulate the input file
    test_years = [10]
    input_df = pd.read_csv("../database/final/players_teams.csv")
    input_df = input_df[["playerID", "year", "tmID", "stint"]]
    input_df = input_df[input_df["year"].isin(test_years)]
    input_df = input_df.reset_index(drop=True)
    print("Input loaded!")

    ps_output = player_stats(input_df, test_years)      # Step 1: Generate player stats from teammates
    ts_output = team_stats(ps_output, test_years)       # Step 2: Generate team stats from player stats
    tr_output = team_ranks(ts_output, test_years)       # Step 3: Generate team ranks from team stats

    print(tr_output)


if __name__ == "__main__":
    main()
