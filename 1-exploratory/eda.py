import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
plt.style.use('ggplot')
sns.set(font_scale=1.2)
pd.set_option('display.max_columns', None)

# Folders
INPUT_FOLDER = '../database/cleaned/'
OUTPUT_ROOT = './output/'

def ensure_output_folder(dataset_name):
    output_folder = os.path.join(OUTPUT_ROOT, dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def save_txt(output_folder, filename, content):
    path = os.path.join(output_folder, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def analyze(df, basename, action):
    output_folder = ensure_output_folder(basename)

    if action == "head" or action == "all":
        save_txt(output_folder, 'head.txt', df.head().to_string())
        print("head saved")

    if action == "describe" or action == "all":
        save_txt(output_folder, 'describe.txt', df.describe().to_string())
        print("describe saved")

    if action == "types" or action == "all":
        with open(os.path.join(output_folder, 'types.txt'), 'w', encoding='utf-8') as f:
            df.info(buf=f)
        print("data types saved")

    if action == "missing" or action == "all":
        save_txt(output_folder, 'missing.txt', df.isna().sum().to_string())
        print("missing values saved")

def main():
    print("Looking for datasets in:", INPUT_FOLDER)
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]

    if not files:
        print("No CSV files found.")
        return

    print("\nAvailable datasets:")
    for idx, f in enumerate(files):
        print(f"[{idx + 1}] {f}")

    choice = input("\nSelect the number of the dataset: ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(files):
        print("Invalid selection.")
        return

    selected_file = files[int(choice) - 1]
    df = pd.read_csv(os.path.join(INPUT_FOLDER, selected_file))
    basename = os.path.splitext(selected_file)[0]

    print("\nAvailable actions:")
    print("[1] head")
    print("[2] describe")
    print("[3] types")
    print("[4] missing")
    print("[5] all")

    action_map = {
        "1": "head",
        "2": "describe",
        "3": "types",
        "4": "missing",
        "5": "all"
    }

    action_choice = input("Select an action: ").strip()
    action = action_map.get(action_choice)

    if not action:
        print("Invalid action.")
        return

    print(f"\nAnalyzing '{selected_file}' with action '{action}'...\n")
    analyze(df, basename, action)
    print(f"\nAnalysis completed. Files saved to ./output/{basename}/")

if __name__ == "__main__":
    main()
