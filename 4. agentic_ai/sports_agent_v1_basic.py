"""
Modern AI Pro Sports Agent v1 - BASIC

Learning Goals:
1. Understand API calls and environment variables
2. Display live data with Streamlit
3. Parse JSON responses

This version: Fetches and displays odds for ONE sport from ONE bookmaker
"""

import streamlit as st
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = os.getenv("ODDS_BASE_URL")

# ============== BASIC API FUNCTION ==============

def get_odds(sport_key):
    """
    Fetch odds for a specific sport
    Returns: List of games with odds data
    """
    url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",           # Only US bookmakers
        "markets": "h2h",          # Only moneyline (head-to-head)
        "oddsFormat": "american"   # +150, -200 format
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code}")
        return []

# ============== STREAMLIT UI ==============

def main():
    # Page setup
    st.set_page_config(
        page_title="Sports Agent v1",
        page_icon="🏀",
        layout="wide"
    )

    st.title("🏀 Sports Agent v1 - Basic Odds Display")
    st.caption("Learn: API calls + JSON parsing + Basic UI")

    # Simple sport selector
    st.subheader("Select Sport")
    sport = st.selectbox(
        "Choose a sport",
        options=["NBA Basketball", "NFL Football", "MLB Baseball"],
        help="Pick your favorite sport"
    )

    # Map display name to API key
    sport_keys = {
        "NBA Basketball": "basketball_nba",
        "NFL Football": "americanfootball_nfl",
        "MLB Baseball": "baseball_mlb"
    }
    selected_sport_key = sport_keys[sport]

    # Fetch button
    if st.button("Get Live Odds", type="primary"):
        with st.spinner("Fetching odds from The Odds API..."):
            events = get_odds(selected_sport_key)

        if events:
            st.success(f"✅ Found {len(events)} upcoming games!")

            # Display each game
            for event in events[:5]:  # Show only first 5 games
                st.divider()

                # Game header
                game_time = datetime.fromisoformat(
                    event['commence_time'].replace('Z', '+00:00')
                ).strftime('%B %d at %I:%M %p')

                st.markdown(f"### 🏀 {event['away_team']} @ {event['home_team']}")
                st.caption(f"🕐 {game_time}")

                # Get odds from first bookmaker
                if event.get('bookmakers'):
                    bookmaker = event['bookmakers'][0]
                    st.write(f"**Bookmaker:** {bookmaker['title']}")

                    # Find the moneyline market
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            col1, col2 = st.columns(2)

                            for outcome in market['outcomes']:
                                odds = outcome['price']
                                team_name = outcome['name']

                                # Color code based on favorite/underdog
                                if odds > 0:  # Underdog
                                    color = "green"
                                    prefix = "+"
                                else:  # Favorite
                                    color = "red"
                                    prefix = ""

                                if team_name == event['home_team']:
                                    with col1:
                                        st.markdown(f"**{team_name} (Home)**")
                                        st.markdown(f":{color}[**{prefix}{odds}**]")
                                else:
                                    with col2:
                                        st.markdown(f"**{team_name} (Away)**")
                                        st.markdown(f":{color}[**{prefix}{odds}**]")
                else:
                    st.warning("No odds available for this game")

            # Show raw data for learning
            with st.expander("🔍 View Raw API Response (for learning)"):
                st.json(events[0] if events else {})
                st.caption("This is the actual JSON data from The Odds API")

        else:
            st.warning("⚠️ No games found. Try another sport!")

    # Educational info
    st.sidebar.markdown("### 📚 How This Works")
    st.sidebar.markdown("""
    **Step 1:** Load API key from .env file

    **Step 2:** Make HTTP GET request to The Odds API

    **Step 3:** Parse JSON response

    **Step 4:** Display data in Streamlit

    **Key Concepts:**
    - Environment variables (API keys)
    - REST API calls
    - JSON data structures
    - Streamlit UI basics
    """)

    st.sidebar.markdown("### 🎯 What's an API?")
    st.sidebar.info(
        "An API (Application Programming Interface) lets your code "
        "request data from another service. Here we're asking "
        "The Odds API for current betting odds."
    )

if __name__ == "__main__":
    main()
