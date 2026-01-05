"""
IPL Oracle - Model Engine
High-Performance Prediction & Betting Engine

This module contains:
- Data cleaning and team name normalization
- Dynamic Elo rating system
- Feature engineering (form guide, home advantage, toss factor)
- Random Forest classifier for match prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pickle
from pathlib import Path


# ============================================================================
# TEAM NAME NORMALIZATION
# ============================================================================

TEAM_NAME_MAPPING = {
    # Delhi franchise
    "Delhi Daredevils": "Delhi Capitals",
    "Delhi Capitals": "Delhi Capitals",

    # Punjab franchise
    "Kings XI Punjab": "Punjab Kings",
    "Punjab Kings": "Punjab Kings",

    # Bangalore franchise
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",

    # Hyderabad franchise (Deccan Chargers -> Sunrisers Hyderabad)
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",

    # Teams with unchanged names
    "Mumbai Indians": "Mumbai Indians",
    "Chennai Super Kings": "Chennai Super Kings",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Rajasthan Royals": "Rajasthan Royals",
    "Gujarat Titans": "Gujarat Titans",
    "Lucknow Super Giants": "Lucknow Super Giants",

    # Defunct teams
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Gujarat Lions": "Gujarat Lions",
    "Pune Warriors": "Pune Warriors",
    "Kochi Tuskers Kerala": "Kochi Tuskers Kerala",
}

# Current active IPL teams (2024)
ACTIVE_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

# Home cities for teams
TEAM_HOME_CITIES = {
    "Chennai Super Kings": ["Chennai"],
    "Delhi Capitals": ["Delhi"],
    "Gujarat Titans": ["Ahmedabad"],
    "Kolkata Knight Riders": ["Kolkata"],
    "Lucknow Super Giants": ["Lucknow"],
    "Mumbai Indians": ["Mumbai"],
    "Punjab Kings": ["Chandigarh", "Mohali", "Dharamsala"],
    "Rajasthan Royals": ["Jaipur"],
    "Royal Challengers Bengaluru": ["Bangalore", "Bengaluru"],
    "Sunrisers Hyderabad": ["Hyderabad"],
    # Defunct teams
    "Rising Pune Supergiant": ["Pune"],
    "Gujarat Lions": ["Rajkot"],
    "Pune Warriors": ["Pune"],
    "Kochi Tuskers Kerala": ["Kochi"],
}


def normalize_team_name(team_name: str) -> str:
    """Normalize team name to current franchise name."""
    if pd.isna(team_name):
        return None
    return TEAM_NAME_MAPPING.get(team_name, team_name)


# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

def load_and_clean_data(filepath: str = None) -> pd.DataFrame:
    """
    Load IPL matches data and perform cleaning:
    - Normalize team names
    - Drop matches with no result
    - Convert date to datetime
    """
    if filepath is None:
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent / "archive" / "Datasets" / "matches_2008-2024.csv",
            Path(__file__).parent / "archive" / "matches_2008-2024.csv",
            Path("archive/Datasets/matches_2008-2024.csv"),
            Path("archive/matches_2008-2024.csv"),
        ]
        for path in possible_paths:
            if path.exists():
                filepath = str(path)
                break
        if filepath is None:
            raise FileNotFoundError("Could not find matches_2008-2024.csv")

    df = pd.read_csv(filepath)

    # Drop matches with no result
    df = df[df['winner'].notna()]
    df = df[~df['result'].isin(['no result', 'NA', 'tie'])]

    # Normalize team names
    df['team1'] = df['team1'].apply(normalize_team_name)
    df['team2'] = df['team2'].apply(normalize_team_name)
    df['winner'] = df['winner'].apply(normalize_team_name)
    df['toss_winner'] = df['toss_winner'].apply(normalize_team_name)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    return df


# ============================================================================
# ELO RATING SYSTEM
# ============================================================================

class EloRatingSystem:
    """
    Dynamic Elo rating system for IPL teams.

    Starting rating: 1500
    K-factor: 32 (standard for competitive play)
    """

    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}
        self.rating_history = []  # Track rating over time

    def get_rating(self, team: str) -> float:
        """Get current rating for a team (initialize if new)."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A against team B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winner: str, loser: str, date: pd.Timestamp = None):
        """Update ratings after a match."""
        rating_winner = self.get_rating(winner)
        rating_loser = self.get_rating(loser)

        expected_winner = self.expected_score(rating_winner, rating_loser)
        expected_loser = 1 - expected_winner

        # Winner gets 1, loser gets 0
        new_rating_winner = rating_winner + self.k_factor * (1 - expected_winner)
        new_rating_loser = rating_loser + self.k_factor * (0 - expected_loser)

        self.ratings[winner] = new_rating_winner
        self.ratings[loser] = new_rating_loser

        # Record history
        if date is not None:
            self.rating_history.append({
                'date': date,
                'team': winner,
                'rating': new_rating_winner,
                'match_result': 'win'
            })
            self.rating_history.append({
                'date': date,
                'team': loser,
                'rating': new_rating_loser,
                'match_result': 'loss'
            })

    def get_history_df(self) -> pd.DataFrame:
        """Get rating history as DataFrame."""
        return pd.DataFrame(self.rating_history)

    def get_win_probability(self, team_a: str, team_b: str) -> float:
        """Get probability of team_a winning against team_b."""
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        return self.expected_score(rating_a, rating_b)


def calculate_all_elo_ratings(df: pd.DataFrame) -> tuple:
    """
    Process all matches and calculate Elo ratings.
    Returns (elo_system, df_with_elo).
    """
    elo = EloRatingSystem()

    # Lists to store Elo ratings for each match
    team1_elos = []
    team2_elos = []
    elo_diffs = []

    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        winner = row['winner']
        date = row['date']

        # Get ratings BEFORE the match
        elo1 = elo.get_rating(team1)
        elo2 = elo.get_rating(team2)

        team1_elos.append(elo1)
        team2_elos.append(elo2)
        elo_diffs.append(elo1 - elo2)

        # Update ratings AFTER the match
        loser = team2 if winner == team1 else team1
        elo.update_ratings(winner, loser, date)

    df = df.copy()
    df['team1_elo'] = team1_elos
    df['team2_elo'] = team2_elos
    df['elo_diff'] = elo_diffs

    return elo, df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def is_home_match(team: str, city: str) -> bool:
    """Check if team is playing at home."""
    if pd.isna(city) or team not in TEAM_HOME_CITIES:
        return False
    home_cities = TEAM_HOME_CITIES[team]
    return any(home_city.lower() in city.lower() for home_city in home_cities)


def calculate_form_guide(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculate win rate in last N matches for each team.
    """
    df = df.copy()

    # Track results for each team
    team_results = {}
    team1_forms = []
    team2_forms = []

    for idx, row in df.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        winner = row['winner']

        # Initialize if needed
        if team1 not in team_results:
            team_results[team1] = []
        if team2 not in team_results:
            team_results[team2] = []

        # Get form BEFORE the match
        form1 = np.mean(team_results[team1][-window:]) if team_results[team1] else 0.5
        form2 = np.mean(team_results[team2][-window:]) if team_results[team2] else 0.5

        team1_forms.append(form1)
        team2_forms.append(form2)

        # Update results AFTER the match
        team_results[team1].append(1 if winner == team1 else 0)
        team_results[team2].append(1 if winner == team2 else 0)

    df['team1_form'] = team1_forms
    df['team2_form'] = team2_forms

    return df


def calculate_venue_stats(df: pd.DataFrame) -> dict:
    """
    Calculate venue statistics:
    - Average score (from target_runs)
    - Chase win percentage
    """
    venue_stats = {}

    for venue in df['venue'].unique():
        venue_matches = df[df['venue'] == venue]

        # Average score
        avg_score = venue_matches['target_runs'].mean()
        if pd.isna(avg_score):
            avg_score = 160  # Default average

        # Chase win percentage (winner decided by "wickets" means chasing team won)
        total_matches = len(venue_matches)
        chase_wins = len(venue_matches[venue_matches['result'] == 'wickets'])
        chase_win_pct = chase_wins / total_matches if total_matches > 0 else 0.5

        venue_stats[venue] = {
            'avg_score': avg_score,
            'chase_win_pct': chase_win_pct,
            'total_matches': total_matches,
            'defend_win_pct': 1 - chase_win_pct
        }

    return venue_stats


def calculate_toss_venue_advantage(df: pd.DataFrame) -> dict:
    """
    Calculate how often toss winner wins at each venue.
    """
    toss_stats = {}

    for venue in df['venue'].unique():
        venue_matches = df[df['venue'] == venue]
        toss_matches = venue_matches[venue_matches['toss_winner'] == venue_matches['winner']]

        total = len(venue_matches)
        toss_wins = len(toss_matches)

        toss_stats[venue] = {
            'toss_win_match_win_pct': toss_wins / total if total > 0 else 0.5,
            'total_matches': total
        }

    return toss_stats


def calculate_head_to_head(df: pd.DataFrame) -> dict:
    """Calculate head-to-head record between all team pairs."""
    h2h = {}

    for _, row in df.iterrows():
        team1, team2, winner = row['team1'], row['team2'], row['winner']

        # Create sorted key for consistency
        key = tuple(sorted([team1, team2]))

        if key not in h2h:
            h2h[key] = {key[0]: 0, key[1]: 0, 'total': 0}

        h2h[key][winner] += 1
        h2h[key]['total'] += 1

    return h2h


def engineer_features(df: pd.DataFrame, venue_stats: dict = None) -> pd.DataFrame:
    """
    Create all features for the model:
    - Elo ratings (already added)
    - Form guide
    - Home advantage
    - Toss features
    - Venue stats
    """
    df = df.copy()

    # Calculate form guide
    df = calculate_form_guide(df)

    # Home advantage
    df['team1_is_home'] = df.apply(
        lambda x: 1 if is_home_match(x['team1'], x['city']) else 0, axis=1
    )
    df['team2_is_home'] = df.apply(
        lambda x: 1 if is_home_match(x['team2'], x['city']) else 0, axis=1
    )

    # Toss features
    df['toss_winner_is_team1'] = (df['toss_winner'] == df['team1']).astype(int)
    df['toss_decision_bat'] = (df['toss_decision'] == 'bat').astype(int)

    # Venue average score
    if venue_stats is None:
        venue_stats = calculate_venue_stats(df)

    df['venue_avg_score'] = df['venue'].map(
        lambda x: venue_stats.get(x, {}).get('avg_score', 160)
    )
    df['venue_chase_win_pct'] = df['venue'].map(
        lambda x: venue_stats.get(x, {}).get('chase_win_pct', 0.5)
    )

    # Target variable: Did team1 win?
    df['team1_win'] = (df['winner'] == df['team1']).astype(int)

    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

FEATURE_COLUMNS = [
    'team1_elo', 'team2_elo', 'elo_diff',
    'team1_form', 'team2_form',
    'team1_is_home', 'team2_is_home',
    'toss_winner_is_team1', 'toss_decision_bat',
    'venue_avg_score', 'venue_chase_win_pct'
]


class IPLPredictor:
    """
    IPL Match Prediction Model.
    Uses Gradient Boosting with Elo-based features and proper regularization.
    """

    def __init__(self, model_type: str = 'logistic'):
        self.model_type = model_type
        self.scaler = StandardScaler()

        # Logistic regression generalizes best for sports prediction
        if model_type == 'logistic':
            self.model = LogisticRegression(
                C=0.5,  # Moderate regularization
                max_iter=1000,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                min_samples_split=30,
                min_samples_leaf=15,
                random_state=42,
                n_jobs=-1
            )

        self.elo_system = None
        self.venue_stats = None
        self.toss_venue_stats = None
        self.h2h_stats = None
        self.is_trained = False
        self.train_accuracy = None
        self.test_accuracy = None
        self.test_log_loss = None
        self.feature_importance = None

    def train(self, df: pd.DataFrame, test_year: int = 2023):
        """
        Train the model using chronological split.
        Train: all matches before test_year
        Test: matches from test_year onwards
        """
        # Calculate Elo ratings
        self.elo_system, df = calculate_all_elo_ratings(df)

        # Calculate venue stats from training data only
        train_mask = df['date'].dt.year < test_year
        train_df = df[train_mask]

        self.venue_stats = calculate_venue_stats(train_df)
        self.toss_venue_stats = calculate_toss_venue_advantage(train_df)
        self.h2h_stats = calculate_head_to_head(df)  # Use all data for H2H

        # Engineer features
        df = engineer_features(df, self.venue_stats)

        # Split data
        X_train = df[train_mask][FEATURE_COLUMNS].copy()
        y_train = df[train_mask]['team1_win']

        test_mask = df['date'].dt.year >= test_year
        X_test = df[test_mask][FEATURE_COLUMNS].copy()
        y_test = df[test_mask]['team1_win']

        # Scale features for better convergence
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        test_proba = self.model.predict_proba(X_test_scaled)

        self.train_accuracy = accuracy_score(y_train, train_pred)
        self.test_accuracy = accuracy_score(y_test, test_pred)
        self.test_log_loss = log_loss(y_test, test_proba)

        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))

        self.is_trained = True

        return {
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'test_log_loss': self.test_log_loss,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

    def predict_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: str,
        toss_decision: str = 'field'
    ) -> dict:
        """
        Predict outcome of a match.
        Returns probabilities for both teams.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Normalize team names
        team1 = normalize_team_name(team1)
        team2 = normalize_team_name(team2)
        toss_winner = normalize_team_name(toss_winner)

        # Get Elo ratings
        elo1 = self.elo_system.get_rating(team1)
        elo2 = self.elo_system.get_rating(team2)

        # Get venue stats
        venue_avg = self.venue_stats.get(venue, {}).get('avg_score', 160)
        venue_chase = self.venue_stats.get(venue, {}).get('chase_win_pct', 0.5)

        # Build feature vector
        features = pd.DataFrame([{
            'team1_elo': elo1,
            'team2_elo': elo2,
            'elo_diff': elo1 - elo2,
            'team1_form': 0.5,  # Default form for new predictions
            'team2_form': 0.5,
            'team1_is_home': 1 if is_home_match(team1, venue) else 0,
            'team2_is_home': 1 if is_home_match(team2, venue) else 0,
            'toss_winner_is_team1': 1 if toss_winner == team1 else 0,
            'toss_decision_bat': 1 if toss_decision == 'bat' else 0,
            'venue_avg_score': venue_avg,
            'venue_chase_win_pct': venue_chase
        }])

        # Predict
        features_scaled = self.scaler.transform(features[FEATURE_COLUMNS])
        proba = self.model.predict_proba(features_scaled)[0]

        # proba[1] is probability of team1 winning
        team1_prob = proba[1]
        team2_prob = proba[0]

        return {
            'team1': team1,
            'team2': team2,
            'team1_probability': team1_prob,
            'team2_probability': team2_prob,
            'team1_elo': elo1,
            'team2_elo': elo2,
            'venue': venue,
            'predicted_winner': team1 if team1_prob > 0.5 else team2
        }

    def calculate_expected_value(
        self,
        team1_prob: float,
        team1_odds: float,
        team2_odds: float
    ) -> dict:
        """
        Calculate Expected Value for betting.

        EV = (Probability * Potential Profit) - (1 - Probability) * Stake
        For decimal odds: EV = (Probability * (Odds - 1)) - (1 - Probability)
        """
        team2_prob = 1 - team1_prob

        # Calculate EV for each team
        team1_ev = (team1_prob * (team1_odds - 1)) - (1 - team1_prob)
        team2_ev = (team2_prob * (team2_odds - 1)) - (1 - team2_prob)

        # Calculate implied probabilities from odds
        team1_implied = 1 / team1_odds
        team2_implied = 1 / team2_odds

        return {
            'team1_ev': team1_ev,
            'team2_ev': team2_ev,
            'team1_value_bet': team1_ev > 0,
            'team2_value_bet': team2_ev > 0,
            'team1_implied_prob': team1_implied,
            'team2_implied_prob': team2_implied,
            'team1_edge': team1_prob - team1_implied,
            'team2_edge': team2_prob - team2_implied
        }

    def save(self, filepath: str = 'ipl_model.pkl'):
        """Save the trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'elo_system': self.elo_system,
                'venue_stats': self.venue_stats,
                'toss_venue_stats': self.toss_venue_stats,
                'h2h_stats': self.h2h_stats,
                'is_trained': self.is_trained,
                'train_accuracy': self.train_accuracy,
                'test_accuracy': self.test_accuracy,
                'test_log_loss': self.test_log_loss,
                'feature_importance': self.feature_importance
            }, f)

    def load(self, filepath: str = 'ipl_model.pkl'):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data.get('scaler', StandardScaler())
            self.elo_system = data['elo_system']
            self.venue_stats = data['venue_stats']
            self.toss_venue_stats = data['toss_venue_stats']
            self.h2h_stats = data['h2h_stats']
            self.is_trained = data['is_trained']
            self.train_accuracy = data['train_accuracy']
            self.test_accuracy = data['test_accuracy']
            self.test_log_loss = data['test_log_loss']
            self.feature_importance = data.get('feature_importance')


# ============================================================================
# HELPER FUNCTIONS FOR UI
# ============================================================================

def get_all_teams(df: pd.DataFrame) -> list:
    """Get list of all teams in the dataset."""
    teams = set(df['team1'].unique()) | set(df['team2'].unique())
    return sorted([t for t in teams if t is not None])


def get_all_venues(df: pd.DataFrame) -> list:
    """Get list of all venues in the dataset."""
    return sorted(df['venue'].unique())


def get_elo_trajectory(elo_system: EloRatingSystem, team: str = None) -> pd.DataFrame:
    """Get Elo rating trajectory for plotting."""
    history_df = elo_system.get_history_df()
    if team:
        history_df = history_df[history_df['team'] == team]
    return history_df


def get_venue_insights(venue_stats: dict, toss_stats: dict) -> pd.DataFrame:
    """Compile venue insights into a DataFrame."""
    data = []
    for venue, stats in venue_stats.items():
        toss_info = toss_stats.get(venue, {})
        data.append({
            'venue': venue,
            'avg_score': stats['avg_score'],
            'chase_win_pct': stats['chase_win_pct'] * 100,
            'defend_win_pct': stats['defend_win_pct'] * 100,
            'toss_advantage': toss_info.get('toss_win_match_win_pct', 0.5) * 100,
            'total_matches': stats['total_matches']
        })
    return pd.DataFrame(data).sort_values('total_matches', ascending=False)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_and_clean_data()
    print(f"Loaded {len(df)} matches")

    # Test different model types
    print("\n" + "="*60)
    print("COMPARING MODEL TYPES")
    print("="*60)

    for model_type in ['logistic', 'gradient_boosting', 'random_forest']:
        print(f"\n--- {model_type.upper()} ---")
        predictor = IPLPredictor(model_type=model_type)
        metrics = predictor.train(df)
        print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"  Test Accuracy:  {metrics['test_accuracy']:.2%}")
        print(f"  Test Log Loss:  {metrics['test_log_loss']:.4f}")

    # Use the best model (logistic regression typically generalizes better)
    print("\n" + "="*60)
    print("FINAL MODEL: Logistic Regression")
    print("="*60)

    predictor = IPLPredictor(model_type='logistic')
    metrics = predictor.train(df)

    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.2%}")
    print(f"  Test Log Loss:  {metrics['test_log_loss']:.4f}")

    # Save model
    predictor.save('ipl_model.pkl')
    print("\nModel saved to ipl_model.pkl")

    # Elo-only baseline
    print("\n--- ELO-ONLY BASELINE ---")
    test_year = 2023
    elo_system, df_elo = calculate_all_elo_ratings(df)
    test_df = df_elo[df_elo['date'].dt.year >= test_year]

    # Predict using Elo probability only
    elo_correct = 0
    for _, row in test_df.iterrows():
        elo1 = row['team1_elo']
        elo2 = row['team2_elo']
        elo_prob = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        elo_pred = row['team1'] if elo_prob > 0.5 else row['team2']
        if elo_pred == row['winner']:
            elo_correct += 1

    elo_accuracy = elo_correct / len(test_df)
    print(f"  Elo-Only Accuracy: {elo_accuracy:.2%}")

    # Example prediction
    print("\nExample Prediction:")
    result = predictor.predict_match(
        team1="Chennai Super Kings",
        team2="Mumbai Indians",
        venue="Wankhede Stadium",
        toss_winner="Mumbai Indians",
        toss_decision="field"
    )
    print(f"  {result['team1']}: {result['team1_probability']:.1%}")
    print(f"  {result['team2']}: {result['team2_probability']:.1%}")
    print(f"  Predicted Winner: {result['predicted_winner']}")
