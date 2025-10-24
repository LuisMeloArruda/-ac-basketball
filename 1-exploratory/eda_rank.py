import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Folders
INPUT_FOLDER = '../database/final/'
OUTPUT_FOLDER = './output_rank/'

# Load base tables
teams = pd.read_csv(os.path.join(INPUT_FOLDER, "teams.csv"))
players = pd.read_csv(os.path.join(INPUT_FOLDER, "players_teams.csv"))
coaches = pd.read_csv(os.path.join(INPUT_FOLDER, "coaches.csv"))
awards = pd.read_csv(os.path.join(INPUT_FOLDER, "awards_players.csv"))

# Rename/Remove attributes
coaches.rename(columns={"won": "coach_wins"}, inplace=True)
coaches.rename(columns={"lost": "coach_losts"}, inplace=True)

# =============== Aggregate players ====================
players_agg = players.groupby(['year', 'tmID']).agg({
    'minutes': 'sum',
    'oRebounds': 'sum',
    'dRebounds': 'sum',
    'GS': 'sum',
    'dq': 'sum'
}).reset_index()

# =============== Aggregate coaches ====================
coaches_agg = coaches.groupby(['year', 'tmID']).agg({
    'coach_wins': 'sum',
    'coach_losts': 'sum',
}).reset_index()

# =============== Aggregate awards ====================
# Count how many awards per team per year (via players)
awards_merged = awards.merge(players[['playerID', 'tmID', 'year']], on=['playerID', 'year'], how='left')
awards_agg = awards_merged.groupby(['year', 'tmID']).size().reset_index(name='num_awards')

# =============== Merge everything ====================
df = teams.merge(players_agg, on=['year', 'tmID'], how='left')
df = df.merge(coaches_agg, on=['year', 'tmID'], how='left')
df = df.merge(awards_agg, on=['year', 'tmID'], how='left')

# Fill NAs (especially for num_awards where some teams might have 0)
df['num_awards'] = df['num_awards'].fillna(0)

# =============== Correlation ====================
numeric_df = df.select_dtypes(include='number')
rank_corr = numeric_df.corr()['rank'].sort_values()

# Save to CSV
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
correlation_path = os.path.join(OUTPUT_FOLDER, "rank_correlation.csv")
rank_corr.to_csv(correlation_path)

print("Saved:", correlation_path)


# =========================================================
# Rank correlation png
# =========================================================

# Load from CSV
rank_corr_series = pd.read_csv(os.path.join(OUTPUT_FOLDER, "rank_correlation.csv"), index_col=0).squeeze("columns")

# Select features (excluding 'rank' itself)
top = rank_corr_series.drop('rank').abs().sort_values(ascending=False).index
rank_corr_top = rank_corr_series[top].sort_values()

# Plot
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid", font_scale=1.0)

bar_colors = rank_corr_top.apply(lambda x: '#2a9d8f' if x < 0 else '#e76f51')

ax = sns.barplot(
    x=rank_corr_top.values,
    y=rank_corr_top.index,
)

# Add horizontal helper lines for each y-tick
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, color='black')
ax.set_axisbelow(True)

plt.title("Top Numeric Features Correlated with Rank", fontsize=14)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.tight_layout()

# Save plot
plot_path = os.path.join(OUTPUT_FOLDER, "rank_correlation.png")
plt.savefig(plot_path)
plt.close()

print("Saved:", plot_path)
