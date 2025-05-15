import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
import os
import time
import uuid
from threading import Timer

def load_data():
    """Load and clean NFL kicker data from data.csv."""
    try:
        data = pd.read_csv('data.csv')
        
        # Calculate success rates for distance ranges (exclude 1-19)
        ranges = ['20_29', '30_39', '40_49', '50_59', '60_Plus']
        for r in ranges:
            data[f'fg_{r}_rate'] = np.where(data[f'FG_{r}_Attempted'] > 0, 
                                           data[f'FG_{r}_Made'] / data[f'FG_{r}_Attempted'], 0)
        
        # Calculate total attempts and success rate
        data['total_attempts'] = sum(data[f'FG_{r}_Attempted'] for r in ranges)
        data['total_made'] = sum(data[f'FG_{r}_Made'] for r in ranges)
        data['overall_rate'] = np.where(data['total_attempts'] > 0,
                                       data['total_made'] / data['total_attempts'], 0)
        
        # Handle missing values
        data = data.fillna(0)
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_model_data(data):
    """Prepare features and labels for prediction model."""
    kicks = []
    range_midpoints = {
        '20_29': [22, 25, 28], '30_39': [32, 35, 38],
        '40_49': [42, 45, 48], '50_59': [52, 55, 58], '60_Plus': [61, 63, 66]
    }
    
    for _, kicker in data.iterrows():
        for r, rate_col in [('20_29', 'fg_20_29_rate'), ('30_39', 'fg_30_39_rate'),
                           ('40_49', 'fg_40_49_rate'), ('50_59', 'fg_50_59_rate'),
                           ('60_Plus', 'fg_60_Plus_rate')]:
            made = int(kicker[f'FG_{r}_Made'])
            missed = int(kicker[f'FG_{r}_Attempted'] - made)
            rate = kicker[rate_col]
            longest = kicker['FG_Longest']
            blocked = kicker['FG_Blocked']
            
            if made + missed > 0:
                made_per_mid = np.random.multinomial(made, [1/3]*3) if made > 0 else [0]*3
                miss_per_mid = np.random.multinomial(missed, [1/3]*3) if missed > 0 else [0]*3
                
                for idx, dist in enumerate(range_midpoints[r]):
                    kicks.extend([(kicker['Player'], dist, rate, longest, blocked, 1) for _ in range(made_per_mid[idx])])
                    kicks.extend([(kicker['Player'], dist, rate, longest, blocked, 0) for _ in range(miss_per_mid[idx])])
    
    kicks_df = pd.DataFrame(kicks, columns=['player', 'distance', 'range_rate', 'longest', 'blocked', 'success'])
    kicks_df['distance_sq'] = kicks_df['distance'] ** 2
    kicks_df['distance_cube'] = kicks_df['distance'] ** 3
    kicks_df['dist_rate'] = kicks_df['distance'] * kicks_df['range_rate']
    
    player_dummies = pd.get_dummies(kicks_df['player'], prefix='player')
    X = pd.concat([kicks_df[['distance', 'distance_sq', 'distance_cube', 'range_rate', 'dist_rate', 'longest', 'blocked']], 
                   player_dummies], axis=1)
    y = kicks_df['success']
    
    # Balance dataset with SMOTE for sparse ranges
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    
    return X, y, X.columns

def train_model(X, y):
    """Train logistic regression model with hyperparameter tuning."""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lr', LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42))
    ])
    
    # Hyperparameter tuning
    param_grid = {'lr__C': [0.01, 0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(model, param_grid, cv=7, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    print(f"Best C: {grid_search.best_params_['lr__C']}")
    cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=7, scoring='accuracy')
    print(f"Model CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return grid_search.best_estimator_

def get_range_stats(data, distance):
    """Get league stats for a distance range."""
    ranges = [(29, '20_29', 22, 25, 28), (39, '30_39', 32, 35, 38),
              (49, '40_49', 42, 45, 48), (59, '50_59', 52, 55, 58), (100, '60_Plus', 61, 63, 66)]
    
    for max_dist, r, m1, m2, m3 in ranges:
        if distance <= max_dist:
            made = data[f'FG_{r}_Made'].sum()
            attempted = data[f'FG_{r}_Attempted'].sum()
            rate = made / attempted if attempted > 0 else 0
            midpoint = m1 if distance <= (m1 + m2) / 2 else (m2 if distance <= (m2 + m3) / 2 else m3)
            return made, attempted, rate, midpoint
    
    return 0, 0, 0, 66

def plot_probability(model, data, player, cols, graph_dir='graphs'):
    """Plot success probability vs. distance and auto-delete after 20-30s."""
    try:
        os.makedirs(graph_dir, exist_ok=True)
        distances = np.arange(20, 72)
        probs = []
        
        for dist in distances:
            prob = predict_success(model, data, player, dist, cols)
            probs.append(0 if isinstance(prob, str) else prob)
        
        plot_data = pd.DataFrame({'Distance': distances, 'Player_Prob': probs})
        
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=plot_data, x='Distance', y='Player_Prob', marker='o', markersize=4, label=player)
        
        for yard in [20, 30, 40, 50, 60]:
            plt.axvline(x=yard, color='gray', linestyle=':', alpha=0.5)
        
        plt.title(f'Field Goal Success for {player}')
        plt.xlabel('Distance (yards)')
        plt.ylabel('Probability')
        plt.ylim(0, 1.02)
        plt.legend()
        
        filename = os.path.join(graph_dir, f'fg_{uuid.uuid4().hex}.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {filename}")
        delete_time = np.random.uniform(20, 30)
        Timer(delete_time, lambda: delete_file(filename)).start()
        print(f"Plot deletes in {delete_time:.1f}s")
        
    except Exception as e:
        print(f"Plot error: {e}")

def delete_file(filename):
    """Delete a file if it exists."""
    try:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted {filename}")
    except Exception as e:
        print(f"Delete error: {e}")

def get_player_stats(data, player, distance):
    """Get player stats for a distance range."""
    ranges = [
        (29, 'fg_20_29_rate', 'FG_20_29_Attempted', 'FG_20_29_Made', (20, 29), '20_29'),
        (39, 'fg_30_39_rate', 'FG_30_39_Attempted', 'FG_30_39_Made', (30, 39), '30_39'),
        (49, 'fg_40_49_rate', 'FG_40_49_Attempted', 'FG_40_49_Made', (40, 49), '40_49'),
        (59, 'fg_50_59_rate', 'FG_50_59_Attempted', 'FG_50_59_Made', (50, 59), '50_59'),
        (100, 'fg_60_Plus_rate', 'FG_60_Plus_Attempted', 'FG_60_Plus_Made', (60, 100), '60_Plus')
    ]
    
    player_data = data[data['Player'] == player]
    for max_dist, rate_col, att_col, made_col, range_tuple, r in ranges:
        if distance <= max_dist:
            attempts = player_data[att_col].values[0]
            made = player_data[made_col].values[0]
            range_rate = player_data[rate_col].values[0]
            longest = player_data['FG_Longest'].values[0]
            blocked = player_data['FG_Blocked'].values[0]
            break
    
    total_attempts = player_data['total_attempts'].values[0]
    total_made = player_data['total_made'].values[0]
    overall_rate = player_data['overall_rate'].values[0]
    
    adj_stats = []
    range_min, range_max = range_tuple
    if range_min > 20:
        lower_key = get_range_key(max(20, range_min - 10))
        if lower_key:
            adj_stats.append((lower_key, abs(distance - ((max(20, range_min - 10) + range_min - 1) / 2))))
    if range_max < 100:
        higher_key = get_range_key(range_max + 1)
        if higher_key:
            adj_stats.append((higher_key, abs(distance - ((range_max + 1 + min(100, range_max + 10)) / 2))))
    
    adj_data = []
    for key, dist_mid in adj_stats:
        rate = player_data[f'fg_{key}_rate'].values[0]
        attempts = player_data[f'FG_{key}_Attempted'].values[0]
        made = player_data[f'FG_{key}_Made'].values[0]
        if attempts >= 3:
            adj_data.append((rate, attempts, dist_mid))
    
    return range_rate, attempts, made, overall_rate, total_attempts, total_made, longest, blocked, adj_data, range_tuple

def get_range_key(distance):
    """Get range key for a distance."""
    ranges = [(29, '20_29'), (39, '30_39'), (49, '40_49'), (59, '50_59'), (100, '60_Plus')]
    for max_dist, key in ranges:
        if distance <= max_dist:
            return key
    return None

def predict_success(model, data, player, distance, cols):
    """Predict field goal success probability with boost for above-average kickers."""
    if player not in data['Player'].values:
        return f"Player '{player}' not found"
    
    range_rate, attempts, made, overall_rate, total_attempts, total_made, longest, blocked, adj_stats, range_tuple = get_player_stats(data, player, distance)
    league_made, league_attempts, league_rate, midpoint = get_range_stats(data, distance)
    
    range_min, range_max = range_tuple
    dist_from_mid = abs(distance - midpoint)
    mid_weight = max(0, 1 - (dist_from_mid / ((range_max - range_min) / 2)) * 0.5)
    
    features = pd.DataFrame(0, index=[0], columns=cols)
    features['distance'] = distance
    features['distance_sq'] = distance ** 2
    features['distance_cube'] = distance ** 3
    features['range_rate'] = range_rate
    features['dist_rate'] = distance * range_rate
    features['longest'] = longest
    features['blocked'] = blocked
    if f'player_{player}' in features.columns:
        features[f'player_{player}'] = 1
    
    # Calculate decay factor
    decay = 1.0 if distance <= 35 else (0.85 - ((distance - 50) / 40) if distance <= 60 else 0.6 - ((distance - 60) / 30))
    
    # Boost for above-average kickers
    perf_boost = 1.0
    if range_rate > league_rate and attempts >= 3:  # Lowered threshold for sensitivity
        perf_diff = min(range_rate - league_rate, 0.2)
        attempt_weight = min(attempts / 8, 1.0)  # Adjusted for smaller samples
        distance_factor = max(0.5, 1 - (distance - 30) / 50)
        perf_boost = 1.0 + (perf_diff * attempt_weight * distance_factor * 1.5)
    elif overall_rate > data['overall_rate'].mean() and total_attempts >= 8:
        perf_diff = min(overall_rate - data['overall_rate'].mean(), 0.15)
        distance_factor = max(0.5, 1 - (distance - 30) / 50)
        perf_boost = 1.0 + (perf_diff * 0.5 * distance_factor)
    
    decay = max(0.1, min(1.0, decay * perf_boost))
    
    prob = 0
    if attempts >= 5:
        try:
            model_prob = model.predict_proba(features)[0][1]
            range_weight = 0.7 if range_rate >= 0.9 and attempts >= 10 else 0.5 * min(attempts / 8, 1)
            prob = range_weight * range_rate + (1 - range_weight) * model_prob
            if range_rate >= 0.95 and distance <= 40:
                prob = max(prob, range_rate * 0.98)
            elif range_rate >= 0.85 and distance <= 50:
                prob = max(prob, range_rate * 0.95)
        except Exception:
            prob = 0
    
    if prob == 0:
        base_prob = range_rate if attempts >= 3 else (overall_rate if total_attempts >= 5 else league_rate)
        if adj_stats:
            adj_sum = sum(rate * min(atts / 5, 1) / (1 + dist / 10) for rate, atts, dist in adj_stats)
            adj_weight = sum(min(atts / 5, 1) / (1 + dist / 10) for _, atts, dist in adj_stats)
            if adj_weight > 0:
                adj_prob = adj_sum / adj_weight
                boundary_factor = max(0, 1 - min(distance - range_min, range_max - distance) / ((range_max - range_min) / 2))
                base_prob = (1 - 0.5 * boundary_factor) * base_prob + 0.5 * boundary_factor * adj_prob  # Increased weight for smoother transitions
        prob = base_prob * mid_weight + base_prob * (1 - mid_weight)
    
    prob = min(prob * perf_boost, 1.0)
    final_prob = (0.85 * prob + 0.15 * league_rate) * decay  # Adjusted blending for robustness
    
    if distance >= 65:
        final_prob = min(final_prob, 0.3)
    
    return max(0.0, min(1.0, final_prob))

def main():
    """Run interactive field goal predictor."""
    data = load_data()
    if data is None:
        print("Failed to load data")
        return
    
    X, y, cols = prepare_model_data(data)
    model = train_model(X, y)
    
    print("\nNFL Kicker Field Goal Predictor")
    print("Available kickers:")
    for player in sorted(data['Player'].values):
        print(f"- {player}")
    
    while True:
        player = input("\nSelect a kicker: ").strip()
        if player in data['Player'].values:
            break
        print(f"Player '{player}' not found.")
    
    while True:
        try:
            distance = int(input("Select distance (20-100 yards): "))
            if 20 <= distance <= 100:
                break
            print("Distance must be 20-100 yards.")
        except ValueError:
            print("Enter a valid number.")
    
    prob = predict_success(model, data, player, distance, cols)
    
    if isinstance(prob, str):
        print(prob)
    else:
        print(f"\n{player} at {distance} yards: {prob:.1%}")
        _, _, league_avg, _ = get_range_stats(data, distance)
        print(f"League average: {league_avg:.1%}")
    
    plot_probability(model, data, player, cols)

if __name__ == "__main__":
    main()