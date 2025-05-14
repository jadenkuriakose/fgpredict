NFL Kicker Field Goal Predictor
Overview
This project predicts the success probability of an NFL kicker making a field goal from a specified distance. It uses a logistic regression model trained on historical kicker performance data, incorporating player-specific and distance-based features. If a kicker lacks attempts in a distance range, the model falls back to a weighted average of the player's performance and the league average.
Requirements

Python 3.8+
Required libraries:
pandas
numpy
scikit-learn



Install dependencies using:
pip install pandas numpy scikit-learn

Setup

Clone or download the repository.
Ensure the data.csv file is in the same directory as nfl_kicker_predictor.py. The CSV should contain NFL kicker data with columns for attempts and made field goals across distance ranges (e.g., FG_1_19_Attempted, FG_1_19_Made, etc.).
Run the script:

python nfl_kicker_predictor.py

Usage

Run the script to start the interactive predictor.
Select a kicker from the displayed list of available players.
Enter a field goal distance (10-100 yards).
View the predicted success probability and the league average for the corresponding distance range.

Example output:
NFL Kicker Field Goal Predictor
Available kickers:
- Player A
- Player B

Select a kicker: Player A
Select a distance between 10-100 yards: 35

Predicted success probability for Player A at 35 yards: 85.2%
League average success rate for this distance range: 80.0%

Data
The model expects a data.csv file with the following structure:

Player: Kicker's name
FG_1_19_Attempted, FG_1_19_Made: Attempts and successful kicks from 1-19 yards
FG_20_29_Attempted, FG_20_29_Made: Attempts and successful kicks from 20-29 yards
FG_30_39_Attempted, FG_30_39_Made: Attempts and successful kicks from 30-39 yards
FG_40_49_Attempted, FG_40_49_Made: Attempts and successful kicks from 40-49 yards
FG_50_59_Attempted, FG_50_59_Made: Attempts and successful kicks from 50-59 yards
FG_60_Plus_Attempted, FG_60_Plus_Made: Attempts and successful kicks from 60+ yards

Notes

If a kicker has no attempts in a distance range, the model uses the league average success rate.
For kickers with limited attempts, predictions blend the player's success rate with the league average, weighted by attempt count.
The model assumes the data in data.csv is accurate and properly formatted.

Future Improvements

Add support for additional features (e.g., weather conditions, home/away status).
Implement model evaluation metrics (e.g., accuracy, ROC-AUC).
Allow batch predictions for multiple kickers and distances.

