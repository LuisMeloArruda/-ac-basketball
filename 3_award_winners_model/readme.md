# WNBA Award Prediction Models

This directory contains three separate machine learning models for predicting
different types of WNBA awards:

## 1. Player Awards (`playerAwards.py`)

Predicts individual player awards based on season performance:

- Most Valuable Player
- Defensive Player of the Year
- Rookie of the Year
- Most Improved Player
- Sixth Woman of the Year
- All-Star Game Most Valuable Player
- WNBA Finals Most Valuable Player
- Kim Perrot Sportsmanship Award

**Key Features:**

- Per-game statistics (PPG, RPG, APG, etc.)
- Shooting percentages and efficiency metrics
- Team performance context
- Year-over-year improvement
- League rankings

## 2. Coach Awards (`coachAwards.py`)

Predicts the Coach of the Year award based on coaching performance:

- Coach of the Year

**Key Features:**

- Win percentage and season record
- Playoff performance
- Team improvement metrics (key factor)
- Conference standings
- Home/away performance
- Historical coaching record

## 3. Team/Decade Awards (`teamAwards.py`)

Predicts special decade team selections (given at milestone years):

- WNBA All-Decade Team
- WNBA All Decade Team Honorable Mention

**Key Features:**

- Career totals and averages
- Peak season performance
- Seasons played
- Playoff appearances and championships
- Individual awards won
- Team success metrics

# Usage

## Training and Evaluation

Each model can be run independently:

```bash
# Player awards
python playerAwards.py

# Coach awards
python coachAwards.py

# Team/decade awards
python teamAwards.py
```
