"""
IPL Oracle - Decision Support System for IPL Match Prediction & Betting

A professional-grade Streamlit dashboard featuring:
- Match Simulator with Win Probability predictions
- Value Finder for betting Expected Value calculations
- Team Analysis with Elo trajectory charts
- Venue Insights with chase/defend statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_engine import (
    IPLPredictor,
    load_and_clean_data,
    calculate_all_elo_ratings,
    calculate_venue_stats,
    calculate_toss_venue_advantage,
    calculate_head_to_head,
    get_all_teams,
    get_all_venues,
    get_venue_insights,
    ACTIVE_TEAMS,
    normalize_team_name
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="IPL Oracle",
    page_icon="cricket_bat_and_ball",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .team-prob {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .team-name {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .value-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .value-negative {
        color: #ff4444;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #1e3c72;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3c72;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING AND MODEL INITIALIZATION
# ============================================================================

@st.cache_data
def load_data():
    """Load and cache the match data."""
    return load_and_clean_data()


@st.cache_resource
def initialize_model():
    """Initialize and train the prediction model."""
    df = load_data()
    predictor = IPLPredictor()
    metrics = predictor.train(df)
    return predictor, metrics


@st.cache_data
def get_elo_history(_predictor):
    """Get Elo rating history for all teams."""
    return _predictor.elo_system.get_history_df()


@st.cache_data
def get_cached_venue_stats(_predictor):
    """Get cached venue statistics."""
    return get_venue_insights(_predictor.venue_stats, _predictor.toss_venue_stats)


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render the sidebar with navigation and model info."""
    st.sidebar.markdown("## IPL Oracle")
    st.sidebar.markdown("*Decision Support System*")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Match Simulator", "Team Analysis", "Venue Insights"],
        index=0
    )

    st.sidebar.markdown("---")

    # Model metrics
    if 'metrics' in st.session_state:
        metrics = st.session_state.metrics
        st.sidebar.markdown("### Model Performance")
        st.sidebar.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")
        st.sidebar.metric("Log Loss", f"{metrics['test_log_loss']:.3f}")
        st.sidebar.caption(f"Trained on {metrics['train_samples']} matches")
        st.sidebar.caption(f"Tested on {metrics['test_samples']} matches (2023-24)")

    return page


# ============================================================================
# MATCH SIMULATOR PAGE
# ============================================================================

def render_match_simulator(predictor, df):
    """Render the Match Simulator page."""
    st.markdown('<h1 class="main-header">Match Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict IPL match outcomes with AI-powered analysis</p>', unsafe_allow_html=True)

    # Get available teams and venues
    all_teams = ACTIVE_TEAMS
    all_venues = get_all_venues(df)

    # Match setup
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        team1 = st.selectbox("Team A", all_teams, index=0, key="team1")

    with col2:
        st.markdown("<div style='text-align: center; padding-top: 2rem; font-size: 1.5rem; font-weight: bold;'>VS</div>", unsafe_allow_html=True)

    with col3:
        # Filter out team1 from team2 options
        team2_options = [t for t in all_teams if t != team1]
        team2 = st.selectbox("Team B", team2_options, index=0, key="team2")

    col1, col2, col3 = st.columns(3)

    with col1:
        venue = st.selectbox("Venue", all_venues, index=0)

    with col2:
        toss_winner = st.selectbox("Toss Winner", [team1, team2], index=0)

    with col3:
        toss_decision = st.selectbox("Toss Decision", ["field", "bat"], index=0)

    st.markdown("---")

    # Predict button
    if st.button("Predict Match", type="primary", use_container_width=True):
        with st.spinner("Analyzing match factors..."):
            result = predictor.predict_match(
                team1=team1,
                team2=team2,
                venue=venue,
                toss_winner=toss_winner,
                toss_decision=toss_decision
            )

            st.session_state.prediction = result

    # Display prediction
    if 'prediction' in st.session_state:
        result = st.session_state.prediction

        # Prediction cards
        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            prob1 = result['team1_probability']
            color1 = "#00c853" if prob1 > 0.5 else "#666"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color1}22, {color1}44);
                        padding: 2rem; border-radius: 15px; text-align: center;
                        border: 2px solid {color1};'>
                <div style='font-size: 1.3rem; margin-bottom: 0.5rem;'>{result['team1']}</div>
                <div style='font-size: 3rem; font-weight: bold; color: {color1};'>{prob1:.0%}</div>
                <div style='font-size: 0.9rem; color: #888;'>Elo: {result['team1_elo']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='text-align: center; padding-top: 3rem;'>
                <div style='font-size: 2rem; font-weight: bold; color: #1e3c72;'>VS</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            prob2 = result['team2_probability']
            color2 = "#00c853" if prob2 > 0.5 else "#666"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color2}22, {color2}44);
                        padding: 2rem; border-radius: 15px; text-align: center;
                        border: 2px solid {color2};'>
                <div style='font-size: 1.3rem; margin-bottom: 0.5rem;'>{result['team2']}</div>
                <div style='font-size: 3rem; font-weight: bold; color: {color2};'>{prob2:.0%}</div>
                <div style='font-size: 0.9rem; color: #888;'>Elo: {result['team2_elo']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Winner announcement
        st.markdown(f"""
        <div style='text-align: center; margin: 2rem 0; padding: 1rem;
                    background: linear-gradient(90deg, #1e3c72, #2a5298);
                    border-radius: 10px; color: white;'>
            <span style='font-size: 1.2rem;'>Predicted Winner: </span>
            <span style='font-size: 1.5rem; font-weight: bold;'>{result['predicted_winner']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Value Finder Widget
        st.markdown("---")
        st.markdown("### Value Finder")
        st.markdown("*Enter market odds to find profitable betting opportunities*")

        col1, col2 = st.columns(2)

        with col1:
            team1_odds = st.number_input(
                f"Market Odds for {result['team1']}",
                min_value=1.01,
                max_value=100.0,
                value=1.80,
                step=0.05,
                key="team1_odds"
            )

        with col2:
            team2_odds = st.number_input(
                f"Market Odds for {result['team2']}",
                min_value=1.01,
                max_value=100.0,
                value=2.10,
                step=0.05,
                key="team2_odds"
            )

        # Calculate EV
        ev_result = predictor.calculate_expected_value(
            result['team1_probability'],
            team1_odds,
            team2_odds
        )

        col1, col2 = st.columns(2)

        with col1:
            ev1 = ev_result['team1_ev']
            edge1 = ev_result['team1_edge']
            color = "#00c853" if ev1 > 0 else "#ff4444"
            value_text = "VALUE BET" if ev1 > 0 else "NO VALUE"

            st.markdown(f"""
            <div style='background: #1a1a2e; padding: 1.5rem; border-radius: 10px;
                        border-left: 4px solid {color};'>
                <div style='color: #888; font-size: 0.9rem;'>{result['team1']}</div>
                <div style='color: {color}; font-size: 2rem; font-weight: bold;'>
                    EV: {ev1:+.2%}
                </div>
                <div style='color: #888; font-size: 0.85rem;'>
                    Edge: {edge1:+.1%} | Implied: {ev_result['team1_implied_prob']:.1%}
                </div>
                <div style='margin-top: 0.5rem; padding: 0.25rem 0.5rem;
                            background: {color}33; color: {color};
                            display: inline-block; border-radius: 5px; font-size: 0.8rem;'>
                    {value_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            ev2 = ev_result['team2_ev']
            edge2 = ev_result['team2_edge']
            color = "#00c853" if ev2 > 0 else "#ff4444"
            value_text = "VALUE BET" if ev2 > 0 else "NO VALUE"

            st.markdown(f"""
            <div style='background: #1a1a2e; padding: 1.5rem; border-radius: 10px;
                        border-left: 4px solid {color};'>
                <div style='color: #888; font-size: 0.9rem;'>{result['team2']}</div>
                <div style='color: {color}; font-size: 2rem; font-weight: bold;'>
                    EV: {ev2:+.2%}
                </div>
                <div style='color: #888; font-size: 0.85rem;'>
                    Edge: {edge2:+.1%} | Implied: {ev_result['team2_implied_prob']:.1%}
                </div>
                <div style='margin-top: 0.5rem; padding: 0.25rem 0.5rem;
                            background: {color}33; color: {color};
                            display: inline-block; border-radius: 5px; font-size: 0.8rem;'>
                    {value_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("*Expected Value (EV) > 0 indicates a potentially profitable bet. Edge = Model Probability - Implied Probability from odds.*")


# ============================================================================
# TEAM ANALYSIS PAGE
# ============================================================================

def render_team_analysis(predictor, df):
    """Render the Team Analysis page."""
    st.markdown('<h1 class="main-header">Team Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore team performance, Elo ratings, and head-to-head records</p>', unsafe_allow_html=True)

    # Get Elo history
    elo_history = get_elo_history(predictor)

    # Team selection for Elo trajectory
    st.markdown("### Elo Rating Trajectory")
    st.markdown("*Track the rise and fall of team strength from 2008 to 2024*")

    selected_teams = st.multiselect(
        "Select teams to compare",
        ACTIVE_TEAMS,
        default=["Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders"]
    )

    if selected_teams:
        # Filter history for selected teams
        filtered_history = elo_history[elo_history['team'].isin(selected_teams)]

        # Create Elo trajectory chart
        fig = px.line(
            filtered_history,
            x='date',
            y='rating',
            color='team',
            title='Elo Rating Over Time',
            labels={'date': 'Date', 'rating': 'Elo Rating', 'team': 'Team'}
        )

        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.add_hline(y=1500, line_dash="dash", line_color="gray",
                      annotation_text="Starting Rating (1500)")

        st.plotly_chart(fig, use_container_width=True)

    # Current Elo Rankings
    st.markdown("---")
    st.markdown("### Current Elo Rankings")

    current_ratings = []
    for team in ACTIVE_TEAMS:
        rating = predictor.elo_system.get_rating(team)
        current_ratings.append({'Team': team, 'Elo Rating': rating})

    ratings_df = pd.DataFrame(current_ratings).sort_values('Elo Rating', ascending=False)
    ratings_df['Rank'] = range(1, len(ratings_df) + 1)
    ratings_df = ratings_df[['Rank', 'Team', 'Elo Rating']]

    col1, col2 = st.columns([1, 1])

    with col1:
        # Bar chart
        fig = px.bar(
            ratings_df,
            x='Elo Rating',
            y='Team',
            orientation='h',
            color='Elo Rating',
            color_continuous_scale='Viridis',
            title='Current Team Strength'
        )
        fig.update_layout(height=400, showlegend=False)
        fig.add_vline(x=1500, line_dash="dash", line_color="red",
                      annotation_text="Baseline")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            ratings_df.style.format({'Elo Rating': '{:.0f}'}).background_gradient(
                subset=['Elo Rating'], cmap='Greens'
            ),
            use_container_width=True,
            hide_index=True,
            height=400
        )

    # Head-to-Head Analysis
    st.markdown("---")
    st.markdown("### Head-to-Head Analysis")

    col1, col2 = st.columns(2)

    with col1:
        h2h_team1 = st.selectbox("Select Team 1", ACTIVE_TEAMS, index=0, key="h2h1")

    with col2:
        h2h_options = [t for t in ACTIVE_TEAMS if t != h2h_team1]
        h2h_team2 = st.selectbox("Select Team 2", h2h_options, index=0, key="h2h2")

    # Get H2H stats
    h2h_key = tuple(sorted([h2h_team1, h2h_team2]))
    h2h_record = predictor.h2h_stats.get(h2h_key, {h2h_team1: 0, h2h_team2: 0, 'total': 0})

    wins1 = h2h_record.get(h2h_team1, 0)
    wins2 = h2h_record.get(h2h_team2, 0)
    total = h2h_record.get('total', 0)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(h2h_team1, f"{wins1} wins", f"{wins1/total:.0%}" if total > 0 else "N/A")

    with col2:
        st.metric("Total Matches", total)

    with col3:
        st.metric(h2h_team2, f"{wins2} wins", f"{wins2/total:.0%}" if total > 0 else "N/A")

    if total > 0:
        # H2H bar chart
        fig = go.Figure(data=[
            go.Bar(name=h2h_team1, x=[h2h_team1], y=[wins1], marker_color='#1e3c72'),
            go.Bar(name=h2h_team2, x=[h2h_team2], y=[wins2], marker_color='#764ba2')
        ])
        fig.update_layout(
            title=f'Head-to-Head Record',
            yaxis_title='Wins',
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# VENUE INSIGHTS PAGE
# ============================================================================

def render_venue_insights(predictor, df):
    """Render the Venue Insights page."""
    st.markdown('<h1 class="main-header">Venue Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover which venues favor chasing vs defending teams</p>', unsafe_allow_html=True)

    # Get venue stats
    venue_df = get_cached_venue_stats(predictor)

    # Filter venues with minimum matches
    min_matches = st.slider("Minimum matches at venue", 5, 50, 10)
    filtered_venues = venue_df[venue_df['total_matches'] >= min_matches]

    # Fortress Check - Chase vs Defend
    st.markdown("### Chase vs Defend Analysis")
    st.markdown("*Venues where chasing teams have advantage (>55% chase win rate) are marked as 'Chaser Friendly'*")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Horizontal bar chart for chase win percentage
        fig = px.bar(
            filtered_venues.sort_values('chase_win_pct', ascending=True),
            x='chase_win_pct',
            y='venue',
            orientation='h',
            color='chase_win_pct',
            color_continuous_scale='RdYlGn',
            title='Chase Win Percentage by Venue',
            labels={'chase_win_pct': 'Chase Win %', 'venue': 'Venue'}
        )

        fig.add_vline(x=50, line_dash="dash", line_color="white",
                      annotation_text="50% (Neutral)")

        fig.update_layout(
            height=max(400, len(filtered_venues) * 25),
            coloraxis_colorbar_title='Chase Win %'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Stats summary
        st.markdown("#### Quick Stats")

        chaser_friendly = filtered_venues[filtered_venues['chase_win_pct'] > 55]
        defender_friendly = filtered_venues[filtered_venues['defend_win_pct'] > 55]

        st.metric("Chaser-Friendly Venues", len(chaser_friendly))
        st.metric("Defender-Friendly Venues", len(defender_friendly))

        if len(chaser_friendly) > 0:
            st.markdown("**Top Chasing Venues:**")
            for _, row in chaser_friendly.nlargest(3, 'chase_win_pct').iterrows():
                venue_short = row['venue'][:30] + "..." if len(row['venue']) > 30 else row['venue']
                st.caption(f"- {venue_short}: {row['chase_win_pct']:.0f}%")

        if len(defender_friendly) > 0:
            st.markdown("**Top Defending Venues:**")
            for _, row in defender_friendly.nlargest(3, 'defend_win_pct').iterrows():
                venue_short = row['venue'][:30] + "..." if len(row['venue']) > 30 else row['venue']
                st.caption(f"- {venue_short}: {row['defend_win_pct']:.0f}%")

    # Average Score Analysis
    st.markdown("---")
    st.markdown("### Venue Scoring Analysis")

    fig = px.scatter(
        filtered_venues,
        x='avg_score',
        y='chase_win_pct',
        size='total_matches',
        color='chase_win_pct',
        hover_name='venue',
        color_continuous_scale='RdYlGn',
        title='Average Score vs Chase Win Rate',
        labels={
            'avg_score': 'Average First Innings Score',
            'chase_win_pct': 'Chase Win %',
            'total_matches': 'Matches Played'
        }
    )

    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Toss Advantage
    st.markdown("---")
    st.markdown("### Toss Advantage by Venue")
    st.markdown("*Venues where winning the toss correlates strongly with match wins*")

    fig = px.bar(
        filtered_venues.sort_values('toss_advantage', ascending=True),
        x='toss_advantage',
        y='venue',
        orientation='h',
        color='toss_advantage',
        color_continuous_scale='Blues',
        title='Toss Winner Match Win % by Venue',
        labels={'toss_advantage': 'Toss Winner Win %', 'venue': 'Venue'}
    )

    fig.add_vline(x=50, line_dash="dash", line_color="red",
                  annotation_text="50% (No Advantage)")

    fig.update_layout(
        height=max(400, len(filtered_venues) * 25),
        coloraxis_colorbar_title='Toss Win %'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed venue table
    st.markdown("---")
    st.markdown("### Detailed Venue Statistics")

    display_df = filtered_venues[['venue', 'total_matches', 'avg_score', 'chase_win_pct', 'defend_win_pct', 'toss_advantage']].copy()
    display_df.columns = ['Venue', 'Matches', 'Avg Score', 'Chase Win %', 'Defend Win %', 'Toss Advantage %']

    st.dataframe(
        display_df.style.format({
            'Avg Score': '{:.0f}',
            'Chase Win %': '{:.1f}%',
            'Defend Win %': '{:.1f}%',
            'Toss Advantage %': '{:.1f}%'
        }).background_gradient(subset=['Chase Win %'], cmap='RdYlGn', vmin=30, vmax=70),
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Initialize model on first run
    if 'predictor' not in st.session_state:
        with st.spinner("Initializing IPL Oracle... Training prediction model..."):
            try:
                predictor, metrics = initialize_model()
                st.session_state.predictor = predictor
                st.session_state.metrics = metrics
                st.session_state.df = load_data()
            except FileNotFoundError as e:
                st.error(f"Error loading data: {e}")
                st.stop()

    predictor = st.session_state.predictor
    df = st.session_state.df

    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if page == "Match Simulator":
        render_match_simulator(predictor, df)
    elif page == "Team Analysis":
        render_team_analysis(predictor, df)
    elif page == "Venue Insights":
        render_venue_insights(predictor, df)


if __name__ == "__main__":
    main()
