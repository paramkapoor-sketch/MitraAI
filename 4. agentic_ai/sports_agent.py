"""
Sports Intelligence Agent - Modern AI Pro Workshop
Combines: The Odds API + DuckDuckGo News + LLM Analysis

Features ChatGPT/Claude CAN'T do:
1. Real-time odds from multiple bookmakers
2. Value bet detection (odds discrepancies)
3. Custom analyst personas
4. Source transparency (see raw data)
5. Multi-regional news lens
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


# Pre-built analyst personas
ANALYST_PERSONAS = {
    "ESPN Analyst": "You are an ESPN sports analyst. Be enthusiastic, use sports metaphors, reference historical matchups, and give confident predictions with reasoning.",
    "Vegas Sharp": "You are a professional sports bettor. Focus on line movement, value opportunities, injury impacts, and bankroll management. Be analytical and probability-focused.",
    "Statistical Guru": "You are a sports statistician. Cite specific stats, use advanced metrics, discuss sample sizes, and express uncertainty appropriately.",
    "Casual Fan": "You are a casual sports fan explaining to a friend. Keep it simple, focus on storylines, star players, and make it fun.",
    "Contrarian Analyst": "You are a contrarian analyst. Look for reasons why the favorite might lose, find value in underdogs, and challenge conventional wisdom."
}

# ============== API FUNCTIONS ==============

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_sports():
    """Fetch available sports from The Odds API"""
    url = f"{ODDS_BASE_URL}/sports/?apiKey={ODDS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_odds(sport_key, regions="us", markets="h2h,spreads,totals"):
    """Fetch odds for a specific sport"""
    url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json(), response.headers
    return [], {}

@st.cache_data(ttl=60)
def get_scores(sport_key, days_from=1):
    """Fetch scores for recent games"""
    url = f"{ODDS_BASE_URL}/sports/{sport_key}/scores/"
    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": days_from
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []

def get_news(query, max_results=5):
    """Fetch news using Tavily (primary) with DuckDuckGo fallback"""
    # Try Tavily first
    if TAVILY_API_KEY:
        try:
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily.search(query, max_results=max_results, search_depth="basic")
            # Convert Tavily format to match expected format
            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("content", ""),
                    "source": r.get("url", "")
                })
            return results
        except Exception as e:
            st.warning(f"Tavily error, falling back to DuckDuckGo: {e}")

    # Fallback to DuckDuckGo
    try:
        results = DDGS().news(query, region='us-en', max_results=max_results)
        return list(results)
    except Exception as e:
        st.error(f"News fetch error: {e}")
        return []

def analyze_with_llm(prompt):
    """Get LLM analysis using Groq"""
    try:
        llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=GROQ_API_KEY)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"LLM Error: {e}"

# ============== ANALYSIS FUNCTIONS ==============

def find_value_bets(events):
    """Detect odds discrepancies across bookmakers"""
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
        
        # Find max odds for each team
        if h2h_odds['home'] and h2h_odds['away']:
            best_home = max(h2h_odds['home'], key=lambda x: x['price'])
            best_away = max(h2h_odds['away'], key=lambda x: x['price'])
            worst_home = min(h2h_odds['home'], key=lambda x: x['price'])
            worst_away = min(h2h_odds['away'], key=lambda x: x['price'])
            
            # Calculate spread
            home_spread = best_home['price'] - worst_home['price']
            away_spread = best_away['price'] - worst_away['price']
            
            if home_spread > 15 or away_spread > 15:  # Significant discrepancy
                value_bets.append({
                    'event': f"{away_team} @ {home_team}",
                    'commence_time': event['commence_time'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'best_home': best_home,
                    'worst_home': worst_home,
                    'best_away': best_away,
                    'worst_away': worst_away,
                    'home_spread': home_spread,
                    'away_spread': away_spread
                })
    
    return sorted(value_bets, key=lambda x: max(x['home_spread'], x['away_spread']), reverse=True)

def format_odds_table(events):
    """Format odds data for display"""
    rows = []
    for event in events:
        if not event.get('bookmakers'):
            continue
        
        row = {
            'Game': f"{event['away_team']} @ {event['home_team']}",
            'Time': datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00')).strftime('%m/%d %I:%M %p')
        }
        
        # Get odds from first few bookmakers
        for i, bookmaker in enumerate(event['bookmakers'][:4]):
            book_name = bookmaker['title'][:10]
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == event['home_team']:
                            row[f'{book_name}_Home'] = outcome['price']
                        else:
                            row[f'{book_name}_Away'] = outcome['price']
        rows.append(row)
    return rows

# ============== STREAMLIT UI ==============

def main():
    st.set_page_config(
        page_title="Sports Intelligence Agent",
        page_icon="🏆",
        layout="wide"
    )
    
    st.title("🏆 Sports Intelligence Agent")
    st.caption("Real-time odds + News + AI Analysis | Built for Modern AI Pro Workshop")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Sport selection
        sports = get_sports()
        sport_options = {s['title']: s['key'] for s in sports if s['active']}
        selected_sport_title = st.selectbox(
            "Select Sport",
            options=list(sport_options.keys()),
            index=list(sport_options.keys()).index('NBA') if 'NBA' in sport_options else 0
        )
        selected_sport = sport_options[selected_sport_title]
        
        # Region selection
        regions = st.multiselect(
            "Bookmaker Regions",
            options=['us', 'uk', 'eu', 'au'],
            default=['us'],
            help="Select regions for odds comparison"
        )
        
        st.divider()
        
        # Analyst persona
        st.subheader("🎭 AI Analyst Persona")
        persona = st.selectbox(
            "Select Style",
            options=list(ANALYST_PERSONAS.keys())
        )
        
        # Custom persona option
        custom_persona = st.text_area(
            "Or create custom persona",
            placeholder="You are a cricket expert focusing on IPL...",
            height=80
        )
        
        st.divider()
        
        # API Usage tracking
        st.subheader("📊 API Usage")
        if st.button("Check Quota"):
            _, headers = get_odds(selected_sport, ",".join(regions))
            if headers:
                st.metric("Remaining", headers.get('x-requests-remaining', 'N/A'))
                st.metric("Used", headers.get('x-requests-used', 'N/A'))
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Odds", 
        "💰 Value Finder", 
        "📰 News + Analysis",
        "🔍 Source Data"
    ])
    
    # Tab 1: Live Odds
    with tab1:
        st.header(f"Live Odds: {selected_sport_title}")
        
        with st.spinner("Fetching odds..."):
            events, headers = get_odds(selected_sport, ",".join(regions))
        
        if events:
            st.success(f"Found {len(events)} upcoming games")
            
            for event in events[:10]:  # Show first 10
                with st.expander(
                    f"🏀 {event['away_team']} @ {event['home_team']} | "
                    f"{datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00')).strftime('%m/%d %I:%M %p')}"
                ):
                    cols = st.columns(len(event.get('bookmakers', [])[:6]))
                    
                    for i, bookmaker in enumerate(event.get('bookmakers', [])[:6]):
                        with cols[i]:
                            st.markdown(f"**{bookmaker['title']}**")
                            for market in bookmaker.get('markets', []):
                                if market['key'] == 'h2h':
                                    for outcome in market['outcomes']:
                                        color = "green" if outcome['price'] > 0 else "red"
                                        prefix = "+" if outcome['price'] > 0 else ""
                                        st.markdown(
                                            f"{outcome['name'][:15]}: "
                                            f":{color}[{prefix}{outcome['price']}]"
                                        )
        else:
            st.warning("No events found for this sport/region combination")
    
    # Tab 2: Value Finder
    with tab2:
        st.header("💰 Value Bet Finder")
        st.info("Finds significant odds discrepancies across bookmakers - potential arbitrage opportunities")
        
        if events:
            value_bets = find_value_bets(events)
            
            if value_bets:
                st.success(f"Found {len(value_bets)} potential value opportunities!")
                
                for vb in value_bets[:5]:
                    with st.container():
                        st.subheader(vb['event'])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{vb['home_team']} (Home)**")
                            st.markdown(f"- Best: **{vb['best_home']['price']:+}** @ {vb['best_home']['book']}")
                            st.markdown(f"- Worst: {vb['worst_home']['price']:+} @ {vb['worst_home']['book']}")
                            st.markdown(f"- Spread: **{vb['home_spread']:.0f} points**")
                        
                        with col2:
                            st.markdown(f"**{vb['away_team']} (Away)**")
                            st.markdown(f"- Best: **{vb['best_away']['price']:+}** @ {vb['best_away']['book']}")
                            st.markdown(f"- Worst: {vb['worst_away']['price']:+} @ {vb['worst_away']['book']}")
                            st.markdown(f"- Spread: **{vb['away_spread']:.0f} points**")
                        
                        st.divider()
            else:
                st.info("No significant odds discrepancies found at this time")
        else:
            st.warning("Load odds data first")
    
    # Tab 3: News + Analysis
    with tab3:
        st.header("📰 News Intelligence + AI Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Game selector for analysis
            if events:
                game_options = [f"{e['away_team']} @ {e['home_team']}" for e in events[:10]]
                selected_game = st.selectbox("Select game to analyze", game_options)
                
                # Extract teams
                selected_event = events[game_options.index(selected_game)]
                teams = f"{selected_event['away_team']} {selected_event['home_team']}"
        
        with col2:
            analyze_btn = st.button("🔍 Analyze Game", type="primary", use_container_width=True)
        
        if analyze_btn and events:
            if not GROQ_API_KEY:
                st.error("Please set GROQ_API_KEY in your .env file")
            else:
                # Fetch news
                with st.spinner("Fetching latest news..."):
                    news = get_news(f"{teams} game preview injury news", max_results=8)
                
                # Display news sources
                st.subheader("📰 Source Articles")
                news_text = ""
                for article in news:
                    with st.expander(f"📄 {article.get('title', 'No title')[:60]}..."):
                        st.write(article.get('body', 'No content'))
                        st.caption(f"Source: {article.get('source', 'Unknown')}")
                    news_text += f"Title: {article.get('title')}\nContent: {article.get('body')}\n\n"
                
                # Get odds context
                odds_context = ""
                for bookmaker in selected_event.get('bookmakers', [])[:3]:
                    odds_context += f"\n{bookmaker['title']}:\n"
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                odds_context += f"  {outcome['name']}: {outcome['price']:+}\n"
                
                # Build analysis prompt
                active_persona = custom_persona if custom_persona else ANALYST_PERSONAS[persona]
                
                prompt = f"""
{active_persona}

Analyze this upcoming game: {selected_game}
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
                
                # Get AI analysis
                with st.spinner("Generating AI analysis..."):
                    analysis = analyze_with_llm(prompt)
                
                st.subheader(f"🤖 {persona} Analysis")
                st.markdown(analysis)
                
                # Show transparency
                with st.expander("🔍 View Full Prompt (Transparency)"):
                    st.code(prompt, language="text")
    
    # Tab 4: Source Data
    with tab4:
        st.header("🔍 Raw Source Data")
        st.info("Full transparency - see exactly what data the AI is working with")
        
        if events:
            st.subheader("Raw API Response")
            st.json(events[:3])  # Show first 3 events
            
            st.subheader("Data Summary")
            st.write(f"- Total events: {len(events)}")
            if events:
                st.write(f"- Bookmakers: {len(events[0].get('bookmakers', []))}")
                st.write(f"- Markets available: {[m['key'] for m in events[0].get('bookmakers', [{}])[0].get('markets', [])]}")

if __name__ == "__main__":
    main()
