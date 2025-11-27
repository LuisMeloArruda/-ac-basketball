import pandas as pd
from playerStats import PlayerStats
from teamRanks import TeamRanks
from teamStats import TeamStats


def main():
    # Simulate the input file
    test_years = [10]
    input_df = pd.read_csv("../database/final/players_teams.csv")
    input_df = input_df[["playerID", "year", "tmID", "stint"]]
    input_df = input_df[input_df["year"].isin(test_years)]
    input_df = input_df.reset_index(drop=True)
    print("Input loaded!")

    # Step 1: Generate player stats from teammates
    training_df = pd.read_csv("../database/final/players_teams.csv")
    training_df = training_df[~(training_df["year"].isin(test_years))]
    training_df = training_df.reset_index(drop=True)
    ps_model = PlayerStats(training_df)
    print("PlayerStats model trained!")

    (ps_input, _) = PlayerStats.preprocess(input_df)
    ps_input = PlayerStats.filterFeatures(ps_input)
    ps_output = PlayerStats.generateResult(ps_model.model, ps_input, ps_model.encoders)
    print("PlayerStats results generated!")

    # Step 2: Generate team stats from player stats
    training_df = pd.read_csv("../database/final/teams.csv")
    training_df = training_df[~training_df["year"].isin(test_years)]
    ts_model = TeamStats(training_df)
    print("TeamStats model trained!")

    (ts_input, _) = TeamStats.preprocessInput(ps_output, ts_model.encoders)
    ts_input = TeamStats.filterFeatures(ts_input)
    ts_output = TeamStats.generateResults(ts_model.model, ts_input, ts_model.encoders)
    print("TeamStats results generated!")

    # Step 3: Generate team ranks from team stats
    training_df = pd.read_csv("../database/final/teams.csv")
    training_df = training_df[~training_df["year"].isin(test_years)]
    tr_model = TeamRanks(training_df)
    print("TeamRanks model trained!")

    (tr_input, _) = TeamRanks.preprocess(ts_output)
    tr_input = TeamRanks.filterFeatures(tr_input)
    tr_output = TeamRanks.generateResults(tr_model.model, tr_input, tr_model.encoders)
    print("TeamStats results generated!")

    print(tr_output)


if __name__ == "__main__":
    main()
