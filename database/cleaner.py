import os
import pandas as pd

RAW_FOLDER = "./raw/"
CLEANED_FOLDER = "./cleaned/"
os.makedirs(CLEANED_FOLDER, exist_ok=True)

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

def clean_csv(name):
    file_path = os.path.join(RAW_FOLDER, f"{name}.csv")
    df = pd.read_csv(file_path)

    # Drop specified columns (only if they exist)
    cols = columns_to_remove.get(name, [])
    df = df.drop(columns=[col for col in cols if col in df.columns])

    # Save cleaned version
    out_path = os.path.join(CLEANED_FOLDER, f"{name}.csv")
    df.to_csv(out_path, index=False)
    print(f" Cleaned: {name}.csv → saved to cleaned/")

def main():
    for name in columns_to_remove.keys():
        clean_csv(name)

if __name__ == "__main__":
    main()
