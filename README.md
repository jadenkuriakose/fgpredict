# NFL Kicker Field Goal Predictor

## Overview
This project predicts the success probability of an NFL kicker making a field goal from a specified distance between 20 and 100 yards. It uses a logistic regression model trained on historical kicker performance data, incorporating player-specific features (e.g., success rate, longest kick, blocked kicks) and distance-based features (e.g., distance, polynomial terms). The model boosts predictions for kickers who outperform the league average, blends player and league data for sparse cases, and generates a plot of success probability versus distance. If a kicker lacks attempts in a distance range, the model falls back to a weighted average of the player's overall performance and adjacent range statistics, smoothed with league averages.

## Requirements
- Python 3.8+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - seaborn
  - matplotlib
  - imbalanced-learn

Install dependencies using:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn
```

## Setup
1. Clone or download the repository.
2. Ensure the `data.csv` file is in the same directory as `main.py`. The CSV should contain NFL kicker data with columns for attempts, made field goals, longest kick, and blocked kicks across distance ranges (e.g., `FG_20_29_Attempted`, `FG_20_29_Made`, `FG_Longest`, `FG_Blocked`).
3. Run the script:
```bash
python main.py
```

## Usage
1. Run the script to start the interactive predictor.
2. Select a kicker from the displayed list of available players (e.g., Chris Boswell, Jake Moody).
3. Enter a field goal distance (20-100 yards).
4. View the predicted success probability and the league average for the corresponding distance range.
5. A plot of the kicker’s success probability versus distance (20-71 yards) is generated and auto-deleted after 20-30 seconds.

Example output:
```
NFL Kicker Field Goal Predictor
Available kickers:
- Chris Boswell
- Jake Moody
- ...

Select a kicker: Chris Boswell
Select distance (20-100 yards): 52

Chris Boswell at 52 yards: 92.5%
League average: 63.9%
Plot saved: graphs/fg_abc123.png
Plot deletes in 25.7s
```

## Data
The model expects a `data.csv` file with the following structure:
- `Player`: Kicker's name (e.g., Jake Moody)
- `FG_20_29_Attempted`, `FG_20_29_Made`: Attempts and successful kicks from 20-29 yards
- `FG_30_39_Attempted`, `FG_30_39_Made`: Attempts and successful kicks from 30-39 yards
- `FG_40_49_Attempted`, `FG_40_49_Made`: Attempts and successful kicks from 40-49 yards
- `FG_50_59_Attempted`, `FG_50_59_Made`: Attempts and successful kicks from 50-59 yards
- `FG_60_Plus_Attempted`, `FG_60_Plus_Made`: Attempts and successful kicks from 60+ yards
- `FG_Longest`: Longest successful field goal distance (in yards)
- `FG_Blocked`: Number of blocked field goal attempts

The provided dataset includes 44 kickers with varying attempt counts, from high-volume players (e.g., Brandon Aubrey, 47 attempts) to low-volume players (e.g., Mitch Wishnowsky, 1 attempt).

## Model Details
- **Algorithm**: Logistic regression with polynomial features (degree 2) and hyperparameter tuning (regularization parameter `C` optimized via GridSearchCV).
- **Features**:
  - Distance, distance², distance³ (captures non-linear effects)
  - Range-specific success rate (e.g., `fg_50_59_rate`)
  - Distance × range rate interaction
  - Longest successful kick (`FG_Longest`)
  - Number of blocked kicks (`FG_Blocked`)
  - Player-specific dummy variables
- **Preprocessing**:
  - StandardScaler for feature normalization
  - SMOTE (Synthetic Minority Oversampling Technique) to balance sparse ranges (e.g., 60+ yards)
- **Performance Boost**: Kickers with above-average success rates (e.g., Chris Boswell, 92.9% at 50-59 yards vs. league 63.9%) receive a probability boost proportional to their outperformance, scaled by attempts and reduced at longer distances.
- **Fallback**: For kickers with few attempts (<3 in a range), predictions blend overall success rate, adjacent range stats, and league averages, weighted by distance proximity and attempt count.
- **Output**: Probability (0-100%), league average for the range, and a plot of player-specific success probability (20-71 yards).

## Weaknesses
1. **Sparse Data for Long Distances**:
   - The 60+ yard range has only 7/25 successes league-wide, leading to noisy predictions. SMOTE helps, but synthetic data may not fully capture real-world variability.
   - Example: Brandon Aubrey (2/4 at 60+) may have inflated probabilities due to limited data.
2. **Limited Features**:
   - Excludes contextual factors like weather, wind, stadium (indoor vs. outdoor), or game pressure, which significantly affect field goal success.
   - Blocked kicks (`FG_Blocked`) are included but don’t account for specific game situations (e.g., defensive pressure).
3. **Model Simplicity**:
   - Logistic regression with polynomial features may not capture complex interactions as well as ensemble methods (e.g., XGBoost, Random Forest).
   - Hyperparameter tuning is limited to `C`; other parameters (e.g., solver, polynomial degree) could be explored.
4. **Small Dataset**:
   - Only 44 kickers, with some having few attempts (e.g., Riley Patterson, 7 total), limit model generalization.
   - Cross-validation accuracy (~0.78-0.85) reflects this constraint, with high variance for low-volume kickers.
5. **Performance Boost Sensitivity**:
   - The boost for above-average kickers (e.g., up to 16.8% for Chris Boswell) may overcorrect for small sample sizes, especially with the lowered attempt threshold (3 attempts).
   - Example: Spencer Shrader (2/2 at 40-49) may receive an unrealistic boost despite limited data.

## Trade-offs
1. **SMOTE vs. Data Sparsity**:
   - **Pro**: Balances sparse ranges (e.g., 60+ yards), improving model performance on rare cases.
   - **Con**: Synthetic data may introduce noise, potentially overestimating success for low-volume kickers.
2. **Logistic Regression vs. Complex Models**:
   - **Pro**: Simple, interpretable, and computationally efficient; suitable for small datasets.
   - **Con**: Misses complex patterns that ensemble methods could capture, limiting accuracy (~0.85 max).
3. **Performance Boost vs. Stability**:
   - **Pro**: Rewards elite kickers (e.g., Boswell’s 92.5% at 52 yards vs. league 63.9%), aligning with real-world expectations.
   - **Con**: Risks overconfidence for kickers with few attempts; mitigated by attempt weighting but not eliminated.
4. **Polynomial Features vs. Overfitting**:
   - **Pro**: Captures non-linear distance effects (e.g., sharp drop-off at 50+ yards).
   - **Con**: Degree-2 interactions increase model complexity, risking overfitting on small data; addressed via regularization tuning.
5. **20-100 Yard Range vs. Realism**:
   - **Pro**: Excludes rare 1-19 yard attempts, focusing on relevant NFL scenarios.
   - **Con**: Ignores short kicks, limiting applicability for edge cases (though uncommon in modern NFL).

## Future Improvements
1. **Incorporate Contextual Data**:
   - Add weather (wind speed, temperature), stadium type (indoor/outdoor), and game situation (e.g., clutch moments) to improve accuracy.
   - Requires external data sources, increasing complexity.
2. **Advanced Models**:
   - Experiment with XGBoost or neural networks to capture non-linear patterns and interactions, potentially boosting accuracy to ~0.90.
   - Trade-off: Increased computational cost and need for larger datasets.
3. **Model Evaluation Metrics**:
   - Add ROC-AUC, precision-recall curves, and calibration plots to assess model reliability beyond accuracy.
   - Example: Ensure predicted probabilities (e.g., 90%) align with actual success rates.
4. **Dynamic Attempt Thresholds**:
   - Adjust the performance boost threshold (3 attempts) based on range-specific data density (e.g., higher for 60+ yards).
   - Reduces overboosting for sparse ranges.
5. **Batch Predictions**:
   - Allow predictions for multiple kickers and distances in one run, useful for team strategy or analysis.
   - Requires UI or script modifications.
6. **Temporal Data**:
   - Incorporate recent performance (e.g., last 5 games) to weight current form over career stats.
   - Needs time-series data, complicating preprocessing.

## Usage Notes
- Ensure `data.csv` matches the expected schema (e.g., `FG_50_59_Made`, `FG_Longest`). Missing columns will cause errors.
- Predictions for low-volume kickers (e.g., Mitch Wishnowsky, 1 attempt) rely heavily on league averages, reducing precision.
- The plot excludes league averages for clarity, focusing on player performance; league averages are still reported in the console.
- Model accuracy (~0.78-0.85) is limited by dataset size and feature scope. Use predictions as a guide, not a definitive outcome.

## Example
For Jake Moody at 52 yards:
- Dataset: 4/9 (44.4%) at 50-59 yards, `FG_Longest` = 53, `FG_Blocked` = 0.
- League average: 159/249 ≈ 63.9%.
- Prediction: ~40-50% (no boost, as 44.4% < 63.9%), slightly increased by `longest` = 53.
- For Chris Boswell (13/14 = 92.9% at 50-59, `FG_Longest` = 57): ~85-95% with performance boost.