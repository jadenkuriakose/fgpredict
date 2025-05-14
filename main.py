import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def loadAndCleanData():
    """
    load and clean the NFL kicker dataset from data.csv to inform predictions.
    """
    try:
        kickerData = pd.read_csv('data.csv')
        
        # Calculate success rates for each distance range
        kickerData['fg1To19Rate'] = np.where(kickerData['FG_1_19_Attempted'] > 0, 
                                           kickerData['FG_1_19_Made'] / kickerData['FG_1_19_Attempted'], 0)
        kickerData['fg20To29Rate'] = np.where(kickerData['FG_20_29_Attempted'] > 0, 
                                            kickerData['FG_20_29_Made'] / kickerData['FG_20_29_Attempted'], 0)
        kickerData['fg30To39Rate'] = np.where(kickerData['FG_30_39_Attempted'] > 0, 
                                            kickerData['FG_30_39_Made'] / kickerData['FG_30_39_Attempted'], 0)
        kickerData['fg40To49Rate'] = np.where(kickerData['FG_40_49_Attempted'] > 0, 
                                            kickerData['FG_40_49_Made'] / kickerData['FG_40_49_Attempted'], 0)
        kickerData['fg50To59Rate'] = np.where(kickerData['FG_50_59_Attempted'] > 0, 
                                            kickerData['FG_50_59_Made'] / kickerData['FG_50_59_Attempted'], 0)
        kickerData['fg60PlusRate'] = np.where(kickerData['FG_60_Plus_Attempted'] > 0, 
                                            kickerData['FG_60_Plus_Made'] / kickerData['FG_60_Plus_Attempted'], 0)
        
        # Calculate total attempts and success rate
        kickerData['totalFgAttempted'] = (
            kickerData['FG_1_19_Attempted'] + 
            kickerData['FG_20_29_Attempted'] + 
            kickerData['FG_30_39_Attempted'] + 
            kickerData['FG_40_49_Attempted'] + 
            kickerData['FG_50_59_Attempted'] + 
            kickerData['FG_60_Plus_Attempted']
        )
        kickerData['totalFgMade'] = (
            kickerData['FG_1_19_Made'] + 
            kickerData['FG_20_29_Made'] + 
            kickerData['FG_30_39_Made'] + 
            kickerData['FG_40_49_Made'] + 
            kickerData['FG_50_59_Made'] + 
            kickerData['FG_60_Plus_Made']
        )
        kickerData['overallFgRate'] = np.where(kickerData['totalFgAttempted'] > 0,
                                             kickerData['totalFgMade'] / kickerData['totalFgAttempted'], 0)
        
        return kickerData
    
    except Exception as e:
        print(f"Error loading data.csv: {e}")
        return None

def prepareModelData(kickerData):
    """
    prepare data for prediction model.
    
    returns a tuple of x (features), y (labels), and feature columns
    """
    allKicks = []
    
    for i, kicker in kickerData.iterrows():
        # Add kicks for each distance range
        ranges = [
            ('FG_1_19', 10), ('FG_20_29', 25), ('FG_30_39', 35),
            ('FG_40_49', 45), ('FG_50_59', 55), ('FG_60_Plus', 63)
        ]
        
        for prefix, dist in ranges:
            made = int(kicker[f'{prefix}_Made'])
            missed = int(kicker[f'{prefix}_Attempted'] - made)
            allKicks.extend([(kicker['Player'], dist, 1) for _ in range(made)])
            allKicks.extend([(kicker['Player'], dist, 0) for _ in range(missed)])
    
    kicksDf = pd.DataFrame(allKicks, columns=['player', 'distance', 'success'])
    playerDummies = pd.get_dummies(kicksDf['player'], prefix='player')
    X = pd.concat([kicksDf[['distance']], playerDummies], axis=1)
    y = kicksDf['success']
    
    return X, y, X.columns

def trainPredictionModel(X, y):
    """
    train a random forest classifier model to predict odds of a kick being successfully made from a range.
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    model.fit(X, y)
    return model

def getLeagueAverage(kickerData, distance):
    """
    get league average success rate for a distance range.
    """
    if distance <= 19:
        made, attempted = 'FG_1_19_Made', 'FG_1_19_Attempted'
    elif distance <= 29:
        made, attempted = 'FG_20_29_Made', 'FG_20_29_Attempted'
    elif distance <= 39:
        made, attempted = 'FG_30_39_Made', 'FG_30_39_Attempted'
    elif distance <= 49:
        made, attempted = 'FG_40_49_Made', 'FG_40_49_Attempted'
    elif distance <= 59:
        made, attempted = 'FG_50_59_Made', 'FG_50_59_Attempted'
    else:
        made, attempted = 'FG_60_Plus_Made', 'FG_60_Plus_Attempted'
    
    totalMade = kickerData[made].sum()
    totalAttempted = kickerData[attempted].sum()
    return totalMade / totalAttempted if totalAttempted > 0 else 0

def predictFgSuccess(model, kickerData, playerName, distance, featureCols):
    """
    predict field goal success probability using player stats & trained model. 
    """
    if playerName not in kickerData['Player'].values:
        return f"Player '{playerName}' not found"
    
    # Determine distance range and relevant columns
    if distance <= 19:
        rateCol, attemptsCol = 'fg1To19Rate', 'FG_1_19_Attempted'
    elif distance <= 29:
        rateCol, attemptsCol = 'fg20To29Rate', 'FG_20_29_Attempted'
    elif distance <= 39:
        rateCol, attemptsCol = 'fg30To39Rate', 'FG_30_39_Attempted'
    elif distance <= 49:
        rateCol, attemptsCol = 'fg40To49Rate', 'FG_40_49_Attempted'
    elif distance <= 59:
        rateCol, attemptsCol = 'fg50To59Rate', 'FG_50_59_Attempted'
    else:
        rateCol, attemptsCol = 'fg60PlusRate', 'FG_60_Plus_Attempted'
    
    playerData = kickerData[kickerData['Player'] == playerName]
    attempts = playerData[attemptsCol].values[0]
    
    # Prepare features for model
    playerFeatures = pd.DataFrame(0, index=[0], columns=featureCols)
    playerFeatures['distance'] = distance
    playerCol = f'player_{playerName}'
    if playerCol in playerFeatures.columns:
        playerFeatures[playerCol] = 1
    
    # Use model prediction if sufficient attempts (at least 3)
    if attempts >= 3:
        try:
            probability = model.predict_proba(playerFeatures)[0][1]
            return probability
        except Exception as e:
            print(f"Prediction error: {e}")
    
    # Fallback: Use player's overall stats with minimal league average influence
    print(f"Note: {playerName} has {attempts} attempts in this range. Using player's stats with statistical adjustment.")
    
    # Get player's overall success rate and attempts
    playerOverallRate = playerData['overallFgRate'].values[0]
    totalAttempts = playerData['totalFgAttempted'].values[0]
    
    # Handle case where player has no attempts at all
    if totalAttempts == 0:
        print(f"Warning: {playerName} has no recorded attempts. Using league average with adjustment.")
        leagueRate = getLeagueAverage(kickerData, distance)
        return 0.9 * leagueRate  # Minimal adjustment to avoid overconfidence
    
    # Get player's rate in the specific range (if available)
    playerRangeRate = playerData[rateCol].values[0] if attempts > 0 else playerOverallRate
    
    # Calculate confidence weight based on total attempts (max weight at 10 attempts)
    weight = min(totalAttempts / 10, 1.0)
    
    # Get league average (minimal influence, max 10% when no attempts)
    leagueRate = getLeagueAverage(kickerData, distance)
    leagueWeight = 0.1 if attempts == 0 else 0.05
    
    # Combine player's rate with minimal league average
    if attempts > 0:
        # Use range-specific rate if available, adjusted by overall performance
        adjustedRate = (0.7 * playerRangeRate) + (0.3 * playerOverallRate)
        probability = (weight * adjustedRate) + ((1 - weight) * playerOverallRate) + (leagueWeight * leagueRate)
    else:
        # When no attempts in range, rely on overall player stats with small league influence
        probability = (0.9 * playerOverallRate) + (leagueWeight * leagueRate)
    
    return max(0.0, min(1.0, probability))

def main():
    """main function to run the predictor interactively."""
    # Load and clean data
    kickerData = loadAndCleanData()
    if kickerData is None:
        print("Failed to load data.csv")
        return
    
    # Prepare and train model
    X, y, featureCols = prepareModelData(kickerData)
    model = trainPredictionModel(X, y)
    
    # Display available kickers
    print("\n  NFL Kicker Field Goal Predictor")
    print("Available kickers:")
    for player in sorted(kickerData['Player'].values):
        print(f"- {player}")
    
    # Prompt for kicker name
    while True:
        playerName = input("\nSelect a kicker: ").strip()
        if playerName in kickerData['Player'].values:
            break
        print(f"Player '{playerName}' not found. Please select from the list.")
    
    # Prompt for distance
    while True:
        try:
            distance = int(input("Select a distance between 10-100 yards: "))
            if 10 <= distance <= 100:
                break
            print("Please enter a distance between 10 and 100 yards.")
        except ValueError:
            print("Please enter a valid numeric distance.")
    
    # Make prediction
    probability = predictFgSuccess(model, kickerData, playerName, distance, featureCols)
    
    # Display result
    if isinstance(probability, str):
        print(probability)
    else:
        print(f"\nPredicted success probability for {playerName} at {distance} yards: {probability:.1%}")
        league_avg = getLeagueAverage(kickerData, distance)
        print(f"League average success rate for this distance range: {league_avg:.1%}")

if __name__ == "__main__":
    main()