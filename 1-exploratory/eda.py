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

def plot_boxplot(df, column, output_folder):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=column)
    plt.title(f'Boxplot da coluna "{column}"')
    boxplot_output_folder = os.path.join(output_folder, "boxplots")
    os.makedirs(boxplot_output_folder, exist_ok=True)
    path = os.path.join(boxplot_output_folder, f'boxplot_{column}.png')
    plt.savefig(path)
    plt.close()
    print(f"boxplot_{column}.png saved")

def plot_heatmap(df, output_folder):
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        print("Poucas colunas numéricas para heatmap.")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(1.2 * len(corr.columns), 1.0 * len(corr.columns)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title("Heatmap de Correlação entre Colunas Numéricas")
    path = os.path.join(output_folder, 'heatmap_correlation.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("heatmap_correlation.png saved")

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

    if action == "boxplots" or action == "all":
        numeric_columns = df.select_dtypes(include='number').columns

        if not numeric_columns.empty:
            for column in numeric_columns:
                plot_boxplot(df, column, output_folder)

    if action == "heatmap" or action == "all":
        plot_heatmap(df, output_folder)

def main():
    print("Looking for datasets in:", INPUT_FOLDER)
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]

    if not files:
        print("No CSV files found.")
        return

    print("\nAvailable datasets:")
    for idx, f in enumerate(files):
        print(f"[{idx + 1}] {f}")

    while (True):
        choice = input("\nSelect the number of the dataset: ").strip()
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(files):
            print("Invalid selection.")
        else: break

    selected_file = files[int(choice) - 1]
    df = pd.read_csv(os.path.join(INPUT_FOLDER, selected_file))
    basename = os.path.splitext(selected_file)[0]

    print("\nAvailable actions:")
    print("[1] head")
    print("[2] describe")
    print("[3] types")
    print("[4] missing")
    print("[5] boxplots")
    print("[6] heatmap")
    print("[7] all")

    action_map = {
        "1": "head",
        "2": "describe",
        "3": "types",
        "4": "missing",
        "5": "boxplots",
        "6": "heatmap",
        "7": "all",
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
