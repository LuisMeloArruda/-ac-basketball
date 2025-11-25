import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

df = pd.read_csv("./outputs/predicted_players_stats.csv")

team = df.groupby(["tmID", "year"]).agg({
    "fgAttempted": "sum",
    "fgMade": "sum",
    "ftMade": "sum",
    "points": "sum",
    "oRebounds": "sum",
    "dRebounds": "sum",
    "assists": "sum",
    "steals": "sum",
    "blocks": "sum",
    "turnovers": "sum",
    "PF": "sum",
    "ftAttempted": "sum",
    "threeMade": "sum",
    "dq": "sum"
}).reset_index()

team["o_fgm"] = team["fgMade"]
team["o_fga"] = team["fgAttempted"]
team["o_ftm"] = team["ftMade"]
team["o_fta"] = team["ftAttempted"]
team["o_3pm"] = team["threeMade"]
team["o_oreb"] = team["oRebounds"]
team["o_dreb"] = team["dRebounds"]
team["o_reb"] = team["oRebounds"] + team["dRebounds"]
team["o_asts"] = team["assists"]
team["o_pf"] = team["PF"]
team["o_stl"] = team["steals"]
team["o_to"] = team["turnovers"]
team["o_blk"] = team["blocks"]
team["o_pts"] = team["points"]

cols = [
    "tmID", "year",
    "o_fgm","o_fga","o_ftm","o_fta","o_3pm",
    "o_oreb","o_dreb","o_reb",
    "o_asts","o_pf","o_stl","o_to","o_blk",
    "o_pts"
]

team = team[cols]

team = team.round(2)

df_pred_players = pd.read_csv("./outputs/predicted_players_stats.csv")
target_year = int(df_pred_players["year"].iloc[0])

print("Target year:", target_year)


df_hist = pd.read_csv("../database/final/teams.csv")
df_hist = df_hist[df_hist["year"] != target_year]

o_cols = [
        "o_fgm",
        "o_fga",
        "o_ftm",
        "o_fta",
        "o_3pm",
        "o_oreb",
        "o_dreb",
        "o_reb",
        "o_asts",
        "o_pf",
        "o_stl",
        "o_to",
        "o_blk",
    ]

d_cols = [
        "d_fga",
        "d_ftm",
        "d_3pm",
        "d_3pa",
        "d_oreb",
        "d_dreb",
        "d_asts",
        "d_pf",
        "d_stl",
        "d_to",
        "d_blk",
        "d_pts",
    ]

X = df_hist[["tmID", "year"] + o_cols].copy()
Y = df_hist[d_cols].copy()


le = LabelEncoder()
X["tmID"] = le.fit_transform(X["tmID"].astype(str))

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
)

model.fit(X, Y)


team_encoded = team.copy()
team_encoded["tmID"] = le.transform(team_encoded["tmID"].astype(str))

X_future = team_encoded[["tmID", "year"] + o_cols]

d_pred = model.predict(X_future)

df_def = pd.DataFrame(d_pred, columns=d_cols)
df_def = df_def.round(2)

df_output = pd.concat([team.reset_index(drop=True), df_def], axis=1)

df_output.to_csv("./outputs/predicted_team_stats.csv", index=False)

print("saved ./outputs/predicted_team_stats.csv")