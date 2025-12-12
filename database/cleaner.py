import os
import pandas as pd

RAW_FOLDER = "./merged/"
CLEANED_FOLDER = "./cleaned/"
FINAL_FOLDER = "./final/"
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(FINAL_FOLDER, exist_ok=True)

columns_to_remove = {
    "players_teams": ["lgID"],
    "players": ["collegeOther", "firstseason", "lastseason", "deathDate"],
    "awards_players": ["lgID"],
    "series_post": ["lgIDWinner", "lgIDLoser"],
    "teams_post": ["lgID"],
    "teams": [
        "lgID", "divID", "seeded",
        "tmORB", "tmDRB", "tmTRB",
        "opptmORB", "opptmDRB", "opptmTRB"
    ],
    "coaches": []
}

columns_to_remove_redundancy = {
    "players_teams": [
    "rebounds", "threeAttempted", "PostRebounds",
    "PostfgAttempted", "PostfgAttempted", "PostMinutes",
    "PostthreeAttempted", "PostftAttempted",
    ],
    "teams": [
    "d_fta", "o_3pa", "min", "d_fga", "d_reb"
    ],
}

values_to_rename = {
    "awards_players": {
        "award": {
            "Kim Perrot Sportsmanship": "Kim Perrot Sportsmanship Award"
        }
    }
}

def remove_columns(df, remove, csv_name):
    cols = remove.get(csv_name, [])
    df = df.drop(columns=[col for col in cols if col in df.columns])
    return df
    
def rename_values(df, rename, csv_name):
    if csv_name in rename:
        for col, mapping in rename[csv_name].items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)
    return df

def save_version(df, output_path, csv_name):
    out_path = os.path.join(output_path, f"{csv_name}.csv")
    df.to_csv(out_path, index=False)
    print(f" {csv_name}.csv → saved to {output_path}")

def clean_csv(name):
    file_path = os.path.join(RAW_FOLDER, f"{name}.csv")
    df = pd.read_csv(file_path)

    df = remove_columns(df, columns_to_remove, name);

    if name == "players":
        df = df[
            df["pos"].notna() &
            df["college"].notna() &
            (df["height"] > 0) &
            (df["weight"] > 0)
        ]

    save_version(df, CLEANED_FOLDER, name)

    df = remove_columns(df, columns_to_remove_redundancy, name)
    
    # Rename "wrong" values
    df = rename_values(df, values_to_rename, name)

    save_version(df, FINAL_FOLDER, name)

def main():
    for name in columns_to_remove.keys():
        clean_csv(name)

if __name__ == "__main__":
    main()
