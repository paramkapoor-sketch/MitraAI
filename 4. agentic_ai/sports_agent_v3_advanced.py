"""
Modern AI Pro Sports Agent v3 - ADVANCED
Learning Goals:
1. Detect value bets (odds discrepancies across bookmakers)
2. Use custom AI personas for different analysis styles
3. Combine ALL features: odds + news + analysis + value detection
4. Build a complete agentic system

This is the FULL production version with all concepts combined.
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

# ============== AI PERSONAS (NEW IN V3!) ==============

ANALYST_PERSONAS = {
    "ESPN Analyst": (
        "You are an enthusiastic ESPN sports analyst. Use sports metaphors, "
        "reference historical matchups, and give confident predictions. "
        "Be energetic and engaging."
    ),
    "Vegas Sharp": (
        "You are a professional sports bettor. Focus on line movement, value opportunities, "
        "injury impacts, and bankroll management. Be analytical and probability-focused."
    ),
    "Statistical Guru": (
        "You are a sports statistician. Cite specific stats, use advanced metrics, "
        "discuss sample sizes, and express appropriate uncertainty."
    ),
    "Casual Fan": (
        "You are a casual sports fan explaining to a friend. Keep it simple, "
        "focus on storylines, star players, and make it fun."
    ),
}

# ============== API FUNCTIONS ==============

@st.cache_data(ttl=300)
def get_odds(sport_key, regions="us"):
    """Fetch odds for a specific sport"""
    url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []

def get_news(query, max_results=5):
    """Fetch news using Tavily or DuckDuckGo"""
    if TAVILY_API_KEY:
        try:
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily.search(query, max_results=max_results)
            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("content", ""),
                    "source": r.get("url", "")
                })
            return results
        except Exception as e:
            st.warning(f"Tavily error, using DuckDuckGo: {e}")

    try:
        results = DDGS().news(query, region='us-en', max_results=max_results)
        return list(results)
    except Exception as e:
        st.error(f"News error: {e}")
        return []

def analyze_with_llm(prompt):
    """Get LLM analysis using Groq"""
    try:
        llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"LLM Error: {e}"

# ============== VALUE BET DETECTION (NEW IN V3!) ==============

def find_value_bets(events, min_spread=15):
    """
    Detect odds discrepancies across bookmakers.
    When odds vary significantly, there's a "value" opportunity!

    How it works:
    1. Collect odds from ALL bookmakers for each team
    2. Find best and worst odds for each team
    3. If spread > threshold, it's a value opportunity
    """
    value_bets = []

    for event in events:
        if not event.get('bookmakers'):
            continue

        home_team = event['home_team']
        away_team = event['away_team']

        # Collect h2h odds from all bookmakers
        h2h_odds = {'home': [], 'away': []}

        for bookmaker in event['bookmakers']:
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            h2h_odds['home'].append({
                                'book': bookmaker['title'],
                                'price': outcome['price']
                            })
                        elif outcome['name'] == away_team:
                            h2h_odds['away'].append({
                                'book': bookmaker['title'],
                                'price': outcome['price']
                            })

        # Find max and min odds (this is value!)
        if h2h_odds['home'] and h2h_odds['away']:
            best_home = max(h2h_odds['home'], key=lambda x: x['price'])
            best_away = max(h2h_odds['away'], key=lambda x: x['price'])
            worst_home = min(h2h_odds['home'], key=lambda x: x['price'])
            worst_away = min(h2h_odds['away'], key=lambda x: x['price'])

            home_spread = best_home['price'] - worst_home['price']
            away_spread = best_away['price'] - worst_away['price']

            if home_spread > min_spread or away_spread > min_spread:
                value_bets.append({
                    'event': f"{away_team} @ {home_team}",
                    'home_team': home_team,
                    'away_team': away_team,
                    'best_home': best_home,
                    'worst_home': worst_home,
                    'best_away': best_away,
                    'worst_away': worst_away,
                    'home_spread': home_spread,
                    'away_spread': away_spread,
                    'commence_time': event['commence_time']
                })

    return sorted(
        value_bets,
        key=lambda x: max(x['home_spread'], x['away_spread']),
        reverse=True
    )

# ============== STREAMLIT UI ==============

def main():
    st.set_page_config(
        page_title="Sports Agent v3",
        page_icon="🏆",
        layout="wide"
    )

    st.title("🏆 Sports Agent v3 - Full Intelligence System")
    st.caption("All concepts combined: Odds + News + AI Personas + Value Detection")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        sport = st.selectbox(
            "Select Sport",
            options=["NBA Basketball", "NFL Football", "MLB Baseball"],
            help="Choose which sport to analyze"
        )

        sport_keys = {
            "NBA Basketball": "basketball_nba",
            "NFL Football": "americanfootball_nfl",
            "MLB Baseball": "baseball_mlb"
        }
        selected_sport_key = sport_keys[sport]

        st.divider()

        # Region selection (NEW IN V3!)
        regions = st.multiselect(
            "Bookmaker Regions",
            options=['us', 'uk', 'eu', 'au'],
            default=['us'],
            help="Compare odds across regions"
        )

        st.divider()

        # AI Persona selection (NEW IN V3!)
        st.subheader("🎭 AI Analyst Persona")
        persona = st.selectbox(
            "Choose analysis style",
            options=list(ANALYST_PERSONAS.keys()),
            help="Different personas provide different insights"
        )

        # Custom persona
        custom_persona = st.text_area(
            "Or create your own persona",
            placeholder="You are a cricket expert focusing on...",
            height=80
        )

        st.divider()

        st.markdown("### 📚 What's New in v3?")
        st.markdown("""
        ✨ **AI Personas** - Different analysis styles

        💎 **Value Detection** - Find odds discrepancies

        🌍 **Multi-region** - Compare global bookmakers

        🔧 **Complete system** - All v1 + v2 features
        """)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Odds",
        "💰 Value Finder",
        "🤖 AI Analysis",
        "🔍 Raw Data"
    ])

    # Initialize events in session
    if st.button("Load Data", type="primary", key="load_data"):
        with st.spinner("Fetching odds..."):
            events = get_odds(selected_sport_key, ",".join(regions))
        st.session_state['events'] = events
        st.session_state['sport_key'] = selected_sport_key
        st.success(f"✅ Loaded {len(events)} games!")

    if 'events' not in st.session_state:
        st.info("👆 Click 'Load Data' to start")
        return

    events = st.session_state['events']

    # ============ TAB 1: LIVE ODDS ============
    with tab1:
        st.header(f"Live Odds: {sport}")
        st.write(f"Showing {len(events[:10])} upcoming games across {len(regions)} region(s)")

        for event in events[:10]:
            game_time = datetime.fromisoformat(
                event['commence_time'].replace('Z', '+00:00')
            ).strftime('%m/%d %I:%M %p')

            with st.expander(
                f"🏀 {event['away_team']} @ {event['home_team']} | {game_time}"
            ):
                cols = st.columns(min(4, len(event.get('bookmakers', []))))

                for i, bookmaker in enumerate(event.get('bookmakers', [])[:4]):
                    with cols[i]:
                        st.markdown(f"**{bookmaker['title']}**")
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'h2h':
                                for outcome in market['outcomes']:
                                    odds = outcome['price']
                                    prefix = "+" if odds > 0 else ""
                                    color = "green" if odds > 0 else "red"
                                    st.markdown(
                                        f"{outcome['name'][:15]}: "
                                        f":{color}[**{prefix}{odds}**]"
                                    )

    # ============ TAB 2: VALUE FINDER (NEW IN V3!) ============
    with tab2:
        st.header("💰 Value Bet Finder")
        st.info(
            "Finds significant odds discrepancies across bookmakers. "
            "When different bookmakers offer very different odds for the same outcome, "
            "there's potential value!"
        )

        # Slider to adjust sensitivity
        min_spread = st.slider(
            "Minimum spread to flag",
            min_value=5,
            max_value=50,
            value=15,
            help="Higher = fewer but more obvious opportunities"
        )

        value_bets = find_value_bets(events, min_spread=min_spread)

        if value_bets:
            st.success(f"🎯 Found {len(value_bets)} value opportunities!")

            for vb in value_bets[:5]:
                with st.container():
                    st.subheader(vb['event'])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### {vb['home_team']} (Home)")
                        st.markdown(f"- **Best odds:** {vb['best_home']['price']:+} @ {vb['best_home']['book']}")
                        st.markdown(f"- Worst odds: {vb['worst_home']['price']:+} @ {vb['worst_home']['book']}")
                        st.markdown(f"- **Spread: {vb['home_spread']:.0f} points** 🔥")

                    with col2:
                        st.markdown(f"### {vb['away_team']} (Away)")
                        st.markdown(f"- **Best odds:** {vb['best_away']['price']:+} @ {vb['best_away']['book']}")
                        st.markdown(f"- Worst odds: {vb['worst_away']['price']:+} @ {vb['worst_away']['book']}")
                        st.markdown(f"- **Spread: {vb['away_spread']:.0f} points** 🔥")

                    st.divider()
        else:
            st.info("No significant discrepancies at this threshold")

    # ============ TAB 3: AI ANALYSIS (NEW PERSONAS!) ============
    with tab3:
        st.header("🤖 AI-Powered Game Analysis")

        if events:
            # Game selector
            game_options = [f"{e['away_team']} @ {e['home_team']}" for e in events[:10]]
            selected_game_idx = st.selectbox(
                "Select game to analyze",
                options=range(len(game_options)),
                format_func=lambda i: game_options[i]
            )

            selected_event = events[selected_game_idx]
            teams = f"{selected_event['away_team']} {selected_event['home_team']}"

            if st.button("🔍 Analyze with AI", type="primary"):
                if not GROQ_API_KEY:
                    st.error("❌ Set GROQ_API_KEY in .env")
                    return

                # Step 1: Fetch news
                st.subheader("📰 Step 1: Gathering News")
                with st.spinner("Searching news..."):
                    news = get_news(f"{teams} game preview injury", max_results=5)

                if news:
                    st.success(f"Found {len(news)} articles")
                    news_text = ""
                    for article in news[:3]:
                        with st.expander(f"📄 {article.get('title', 'No title')[:60]}..."):
                            st.write(article.get('body', '')[:300] + "...")
                            st.caption(f"Source: {article.get('source', 'Unknown')}")
                        news_text += f"- {article.get('title')}\n  {article.get('body', '')[:150]}\n\n"

                # Step 2: Get odds context
                st.subheader("📊 Step 2: Current Odds")
                odds_context = ""
                for bookmaker in selected_event.get('bookmakers', [])[:3]:
                    odds_context += f"{bookmaker['title']}:\n"
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                odds_context += f"  {outcome['name']}: {outcome['price']:+}\n"

                st.code(odds_context)

                # Step 3: Select persona (NEW IN V3!)
                st.subheader("🎭 Step 3: Select Analysis Persona")
                active_persona = custom_persona if custom_persona else ANALYST_PERSONAS[persona]
                st.info(f"Using: **{persona}** persona")

                # Step 4: Get AI analysis
                st.subheader("🤖 Step 4: Generating Analysis")

                prompt = f"""
{active_persona}

Analyze this upcoming game: {selected_game_idx + 1}. {selected_event['away_team']} @ {selected_event['home_team']}
Game Time: {selected_event['commence_time']}

CURRENT BETTING ODDS:
{odds_context}

LATEST NEWS & INFORMATION:
{news_text}

Provide:
1. Key factors that will influence this game
2. Your prediction with confidence level
3. Betting recommendation (if appropriate)
4. Any concerns or things to watch

Be specific and reference the actual news/odds data provided.
"""

                with st.spinner("AI is thinking..."):
                    analysis = analyze_with_llm(prompt)

                st.markdown(analysis)

                # Show transparency (NEW IN V3!)
                with st.expander("🔍 View Full Prompt (Prompt Engineering)"):
                    st.code(prompt, language="text")
                    st.caption(
                        "This is the exact prompt sent to the LLM. "
                        "Notice how the persona affects the analysis style!"
                    )

    # ============ TAB 4: RAW DATA ============
    with tab4:
        st.header("🔍 Raw Data (Full Transparency)")
        st.info("See exactly what data the AI is working with")

        if events:
            st.subheader("API Response Sample")
            st.json(events[0])

            st.subheader("Data Summary")
            st.write(f"- Total games: {len(events)}")
            st.write(f"- Bookmakers per game: {len(events[0].get('bookmakers', []))}")
            if events[0].get('bookmakers'):
                markets = [m['key'] for m in events[0]['bookmakers'][0].get('markets', [])]
                st.write(f"- Available markets: {markets}")

    # Educational footer
    st.divider()
    with st.expander("📚 Learning: How Agentic AI Works"):
        st.markdown("""
        ### The Complete Agent Pattern

        This system demonstrates a complete **Agentic AI** agent:

        **1. Data Collection** (Multiple Sources)
        - The Odds API → Real-time betting odds
        - Tavily/DuckDuckGo → Recent news
        - Multiple bookmakers → Price comparison

        **2. Intelligence** (Value Detection)
        - Detects odds discrepancies across bookmakers
        - Identifies arbitrage opportunities
        - Something ChatGPT can't do!

        **3. Analysis** (LLM Integration)
        - Custom personas for different analysis styles
        - Combines live data into prompts
        - Real-time reasoning on fresh information

        **4. Transparency** (Show Your Work)
        - See the raw data
        - View the exact prompts sent
        - Understand how the AI works

        ### Why This Beats ChatGPT

        ✅ **Real-time data** - Latest odds & news
        ✅ **Domain logic** - Value detection algorithm
        ✅ **Fresh context** - News from this week
        ✅ **Custom personalities** - Different analysis styles
        ✅ **Transparent** - See how decisions are made

        ChatGPT has:
        ❌ Stale training data (months/years old)
        ❌ No real-time odds
        ❌ No custom analysis logic
        ❌ Generic responses
        """)

if __name__ == "__main__":
    main()
