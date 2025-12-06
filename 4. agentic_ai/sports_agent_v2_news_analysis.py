"""
Modern AI Pro  Sports Agent v2 - NEWS + AI ANALYSIS
Learning Goals:
1. Combine multiple data sources (Odds + News)
2. Use LLM for intelligent analysis
3. Create prompts that work with live data
4. Build multi-section interfaces

New features vs v1:
- News fetching with Tavily/DuckDuckGo
- LLM integration with Groq
- Combining odds + news for analysis
- Multiple games display
"""

import streamlit as st
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from tavily import TavilyClient

# Load environment variables
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = os.getenv("ODDS_BASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ============== API FUNCTIONS ==============

def get_odds(sport_key):
    """Fetch odds for a specific sport"""
    url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []

def get_news(query, max_results=5):
    """
    Fetch news using Tavily (primary) with DuckDuckGo fallback
    This is how we get real-time information!
    """
    # Try Tavily first (better quality results)
    if TAVILY_API_KEY:
        try:
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily.search(query, max_results=max_results)

            # Convert to consistent format
            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("content", ""),
                    "source": r.get("url", "")
                })
            return results
        except Exception as e:
            st.warning(f"Tavily error: {e}, using DuckDuckGo fallback")

    # Fallback to DuckDuckGo
    try:
        results = DDGS().news(query, region='us-en', max_results=max_results)
        return list(results)
    except Exception as e:
        st.error(f"News error: {e}")
        return []

def analyze_with_llm(prompt):
    """
    Use Groq's LLM to analyze the data
    This is where the "intelligence" happens!
    """
    try:
        llm = ChatGroq(
            model_name="openai/gpt-oss-120b",
            api_key=GROQ_API_KEY
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"LLM Error: {e}"

# ============== STREAMLIT UI ==============

def main():
    st.set_page_config(
        page_title="Sports Agent v2",
        page_icon="🏆",
        layout="wide"
    )

    st.title("🏆 Sports Agent v2 - News + AI Analysis")
    st.caption("Learn: Multi-source data + LLM integration + Prompt engineering")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        sport = st.selectbox(
            "Select Sport",
            options=["NBA Basketball", "NFL Football", "NHL Hockey"]
        )

        sport_keys = {
            "NBA Basketball": "basketball_nba",
            "NFL Football": "americanfootball_nfl",
            "NHL Hockey": "icehockey_nhl"
        }
        selected_sport_key = sport_keys[sport]

        st.divider()

        st.markdown("### 📚 What's New in v2?")
        st.markdown("""
        ✅ News aggregation (Tavily/DDG)

        ✅ LLM analysis with Groq

        ✅ Smart prompting with context

        ✅ Multi-source intelligence
        """)

    # Main content - Two sections
    st.header("📊 Section 1: Live Odds")

    if st.button("Fetch Odds", type="primary"):
        with st.spinner("Loading odds..."):
            events = get_odds(selected_sport_key)

        if events:
            st.success(f"Found {len(events)} upcoming games")

            # Store in session state for later use
            st.session_state['events'] = events
            st.session_state['sport_key'] = selected_sport_key

            # Display odds in a simpler format
            for i, event in enumerate(events[:5]):
                with st.expander(
                    f"🏀 {event['away_team']} @ {event['home_team']}"
                ):
                    # Game time
                    game_time = datetime.fromisoformat(
                        event['commence_time'].replace('Z', '+00:00')
                    ).strftime('%B %d at %I:%M %p')
                    st.caption(f"🕐 {game_time}")

                    # Show odds from available bookmakers
                    if event.get('bookmakers'):
                        cols = st.columns(min(3, len(event['bookmakers'])))

                        for j, bookmaker in enumerate(event['bookmakers'][:3]):
                            with cols[j]:
                                st.markdown(f"**{bookmaker['title']}**")
                                for market in bookmaker.get('markets', []):
                                    if market['key'] == 'h2h':
                                        for outcome in market['outcomes']:
                                            odds = outcome['price']
                                            prefix = "+" if odds > 0 else ""
                                            st.write(f"{outcome['name'][:12]}: {prefix}{odds}")

    st.divider()

    # Section 2: News + AI Analysis
    st.header("🤖 Section 2: AI-Powered Game Analysis")

    if 'events' in st.session_state:
        events = st.session_state['events']

        # Game selector
        game_options = [
            f"{e['away_team']} @ {e['home_team']}"
            for e in events[:5]
        ]
        selected_game = st.selectbox(
            "Select a game to analyze",
            options=game_options
        )

        if st.button("🔍 Analyze This Game", type="primary"):
            if not GROQ_API_KEY:
                st.error("❌ Please set GROQ_API_KEY in your .env file")
                return

            # Get the selected event
            game_index = game_options.index(selected_game)
            selected_event = events[game_index]

            # Extract team names for news search
            teams = f"{selected_event['away_team']} {selected_event['home_team']}"

            # Step 1: Fetch news
            st.subheader("📰 Step 1: Fetching Recent News")
            with st.spinner("Searching news sources..."):
                news = get_news(f"{teams} game preview injury", max_results=5)

            if news:
                st.success(f"✅ Found {len(news)} relevant articles")

                # Display news
                news_text = ""
                for article in news:
                    with st.expander(f"📄 {article.get('title', 'No title')[:50]}..."):
                        st.write(article.get('body', 'No content')[:300] + "...")
                        st.caption(f"Source: {article.get('source', 'Unknown')}")

                    # Build context for LLM
                    news_text += f"- {article.get('title')}\n"
                    news_text += f"  {article.get('body', '')[:200]}\n\n"

                # Step 2: Get odds context
                st.subheader("📊 Step 2: Current Betting Odds")
                odds_context = ""
                if selected_event.get('bookmakers'):
                    for bookmaker in selected_event['bookmakers'][:2]:
                        odds_context += f"{bookmaker['title']}:\n"
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'h2h':
                                for outcome in market['outcomes']:
                                    odds_context += f"  {outcome['name']}: {outcome['price']:+}\n"

                st.code(odds_context)

                # Step 3: Build prompt and get AI analysis
                st.subheader("🤖 Step 3: AI Analysis")

                prompt = f"""
You are a sports analyst. Analyze this upcoming game.

GAME: {selected_game}
TIME: {selected_event['commence_time']}

CURRENT ODDS:
{odds_context}

RECENT NEWS:
{news_text}

Provide a brief analysis covering:
1. Key storylines or factors (injuries, momentum, etc.)
2. What the odds suggest
3. Your prediction

Keep it concise (3-4 paragraphs).
"""

                with st.spinner("Generating AI analysis..."):
                    analysis = analyze_with_llm(prompt)

                st.markdown(analysis)

                # Show the prompt for learning
                with st.expander("🔍 View Prompt (Learn prompt engineering)"):
                    st.code(prompt, language="text")
                    st.caption(
                        "This is the exact prompt sent to the LLM. "
                        "Notice how we combine odds + news into one context!"
                    )

            else:
                st.warning("No news found for this matchup")

    else:
        st.info("👆 Fetch odds first to enable game analysis")

    # Learning sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 💡 Key Concepts")
    st.sidebar.markdown("""
    **Agentic AI** combines:

    1. **Data Sources** (Odds API, News API)
    2. **LLM** (Groq for analysis)
    3. **Prompting** (Combining context)
    4. **Real-time** (Live odds + recent news)

    This is more powerful than ChatGPT because
    we control the data sources and can use
    real-time information!
    """)

if __name__ == "__main__":
    main()
