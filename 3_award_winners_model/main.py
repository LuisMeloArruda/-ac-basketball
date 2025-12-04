from playerAwards import PlayerAwards
from coachAwards import CoachAwards
from teamAwards import TeamAwards

# Initialize models
player_model = PlayerAwards(data_path="../database/final/")
coach_model = CoachAwards(data_path="../database/final/")
team_model = TeamAwards(data_path="../database/final/")

# Load data
player_model.load_data()
coach_model.load_data()
team_model.load_data()

# Train models
player_model.train_all_models()
coach_model.train_all_models()
team_model.train_all_models()

# Predict for a specific year
year = 11
player_predictions = player_model.predict_award_winners(year)
coach_predictions = coach_model.predict_award_winners(year)
team_predictions = team_model.predict_award_winners(year)

print(player_predictions.head())
print(coach_predictions.head())
print(team_predictions.head())