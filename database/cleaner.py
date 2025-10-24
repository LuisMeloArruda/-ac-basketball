import os
import pandas as pd

RAW_FOLDER = "./raw/"
CLEANED_FOLDER = "./cleaned/"
FINAL_FOLDER = "./final/"
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(FINAL_FOLDER, exist_ok=True)

columns_to_remove = {
    "players_teams": ["lgID"],
    "players": ["collegeOther", "firstseason", "lastseason", "deathDate", "birthDate"],
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
    "fgAttempted", "fgMade",
    "rebounds", "threeMade", "PostRebounds",
    "PostfgAttempted", "PostfgMade", "PostMinutes",
    "PostthreeMade", "PostftMade", "ftMade"
    ],
    "teams": [
    "homeL", "awayL", "lost",
    "d_fta", "o_3pa", "min",
    "o_pts", "d_pts", "d_reb"
    ],
}

def remove_columns(df, remove, csv_name):
    cols = remove.get(csv_name, [])
    df = df.drop(columns=[col for col in cols if col in df.columns])
    return df

def save_version(df, output_path, csv_name):
    out_path = os.path.join(output_path, f"{csv_name}.csv")
    df.to_csv(out_path, index=False)
    print(f" {csv_name}.csv → saved to {output_path}")

def clean_csv(name):
    file_path = os.path.join(RAW_FOLDER, f"{name}.csv")
    df = pd.read_csv(file_path)

    # Drop specified columns (only if they exist)
    df = remove_columns(df, columns_to_remove, name);

    # Filters
    if name == "players":
        df = df[
            df["pos"].notna() &
            df["college"].notna() &
            (df["height"] > 0) &
            (df["weight"] > 0)
        ]

    # Save cleaned version
    save_version(df, CLEANED_FOLDER, name)

    # Drop redundant columns
    df = remove_columns(df, columns_to_remove_redundancy, name)

    # Save final version
    save_version(df, FINAL_FOLDER, name)

def main():
    for name in columns_to_remove.keys():
        clean_csv(name)

if __name__ == "__main__":
    main()
