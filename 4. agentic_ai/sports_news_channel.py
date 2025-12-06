"""
🎬 AI Sports News Channel - Modern AI Pro Workshop
Combines: Odds API + News + LLM Analysis + HeyGen Video Avatar

BUILD YOUR OWN ESPN!

Features:
1. Real-time odds from bookmakers
2. News aggregation via DuckDuckGo  
3. LLM-powered analysis (Groq)
4. AI Avatar video generation (HeyGen)
5. Custom anchor personas
"""

import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from tavily import TavilyClient

# Load environment variables
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = os.getenv("ODDS_BASE_URL")
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_BASE_URL = os.getenv("HEYGEN_BASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Sports anchor personas (for both text and video)
ANCHOR_PERSONAS = {
    "🎙️ Classic ESPN Anchor": {
        "prompt": "You are a professional ESPN sports anchor. Be energetic, use dramatic pauses, reference key matchups, and deliver with authority. Keep it under 60 seconds when spoken.",
        "voice_style": "professional, energetic",
        "suggested_avatar": "sports_anchor"
    },
    "📊 Stats Analyst": {
        "prompt": "You are a data-driven sports analyst like Nate Silver. Focus on probabilities, historical trends, and statistical edges. Be precise but accessible. Keep it under 60 seconds.",
        "voice_style": "analytical, measured",
        "suggested_avatar": "business_professional"
    },
    "🔥 Hot Take Host": {
        "prompt": "You are a provocative sports talk host like Stephen A. Smith. Be bold, make predictions, show emotion, but back it up. Keep it under 60 seconds.",
        "voice_style": "passionate, dramatic",
        "suggested_avatar": "energetic_presenter"
    },
    "🏏 Cricket Commentator": {
        "prompt": "You are a legendary cricket commentator. Use cricket terminology, be eloquent, reference the spirit of the game. Perfect for IPL coverage. Keep it under 60 seconds.",
        "voice_style": "refined, enthusiastic",
        "suggested_avatar": "british_presenter"
    },
    "🎯 Betting Sharp": {
        "prompt": "You are a professional sports bettor sharing insights. Focus on line value, market movements, and bankroll advice. Be calculated and strategic. Keep it under 60 seconds.",
        "voice_style": "calm, authoritative",
        "suggested_avatar": "business_casual"
    }
}

# ============== ODDS API FUNCTIONS ==============

@st.cache_data(ttl=300)
def get_sports():
    """Fetch available sports"""
    url = f"{ODDS_BASE_URL}/sports/?apiKey={ODDS_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

@st.cache_data(ttl=60)
def get_odds(sport_key, regions="us"):
    """Fetch odds for a sport"""
    url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json(), response.headers
    except:
        pass
    return [], {}

# ============== NEWS FUNCTIONS ==============

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
        st.warning(f"News fetch limited: {e}")
        return []

# ============== LLM FUNCTIONS ==============

def generate_script(event_data, odds_data, news_data, persona):
    """Generate anchor script using LLM"""
    try:
        llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=GROQ_API_KEY)
        
        prompt = f"""
{persona['prompt']}

You are delivering a sports update segment. Create a script that:
1. Opens with a hook about the upcoming game
2. Covers the betting lines and what they mean
3. Incorporates relevant news/injury updates
4. Closes with your prediction

GAME: {event_data['away_team']} @ {event_data['home_team']}
TIME: {event_data['commence_time']}

BETTING ODDS:
{odds_data}

LATEST NEWS:
{news_data}

Write the script in first person, as if you're speaking directly to camera.
Include natural pauses indicated by "..." 
Keep it conversational and under 150 words (about 60 seconds when spoken).
Do NOT include stage directions or [brackets].
"""
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Script generation error: {e}"

# ============== HEYGEN API FUNCTIONS ==============

def get_heygen_avatars(api_key):
    """Fetch available HeyGen avatars"""
    url = f"{HEYGEN_BASE_URL}/v2/avatars"
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', {}).get('avatars', [])
    except Exception as e:
        st.error(f"Avatar fetch error: {e}")
    return []

def get_heygen_voices(api_key):
    """Fetch available HeyGen voices"""
    url = f"{HEYGEN_BASE_URL}/v2/voices"
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', {}).get('voices', [])
    except Exception as e:
        st.error(f"Voice fetch error: {e}")
    return []

def generate_heygen_video(api_key, script, avatar_id, voice_id, test_mode=True):
    """Generate video using HeyGen API"""
    url = f"{HEYGEN_BASE_URL}/v2/video/generate"
    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "text",
                    "input_text": script,
                    "voice_id": voice_id
                },
                "background": {
                    "type": "color",
                    "value": "#1a1a2e"  # Dark professional background
                }
            }
        ],
        "dimension": {
            "width": 1280,
            "height": 720
        },
        "test": test_mode  # Set to False for production (uses credits)
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', {}).get('video_id')
        else:
            st.error(f"HeyGen API error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Video generation error: {e}")
    return None

def check_video_status(api_key, video_id):
    """Check HeyGen video generation status"""
    url = f"{HEYGEN_BASE_URL}/v1/video_status.get?video_id={video_id}"
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', {})
    except Exception as e:
        st.error(f"Status check error: {e}")
    return {}

# ============== HELPER FUNCTIONS ==============

def format_odds_for_script(event):
    """Format odds data for LLM consumption"""
    odds_text = ""
    for bookmaker in event.get('bookmakers', [])[:3]:
        odds_text += f"\n{bookmaker['title']}:\n"
        for market in bookmaker.get('markets', []):
            if market['key'] == 'h2h':
                for outcome in market['outcomes']:
                    prefix = "+" if outcome['price'] > 0 else ""
                    odds_text += f"  {outcome['name']}: {prefix}{outcome['price']}\n"
            elif market['key'] == 'spreads':
                for outcome in market['outcomes']:
                    prefix = "+" if outcome.get('point', 0) > 0 else ""
                    odds_text += f"  {outcome['name']} spread: {prefix}{outcome.get('point', 'N/A')}\n"
    return odds_text

def format_news_for_script(news_items):
    """Format news for LLM consumption"""
    news_text = ""
    for item in news_items[:5]:
        news_text += f"- {item.get('title', 'No title')}\n"
        news_text += f"  {item.get('body', '')[:200]}...\n\n"
    return news_text

# ============== STREAMLIT UI ==============

def main():
    st.set_page_config(
        page_title="AI Sports News Channel",
        page_icon="🎬",
        layout="wide"
    )
    
    # Custom CSS for news channel feel
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .breaking-news {
        background: #e63946;
        color: white;
        padding: 10px;
        border-radius: 5px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🎬 AI Sports News Channel</h1>
        <p style="color: #888; margin: 0;">Your Personalized ESPN • Powered by Modern AI Pro</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        # Sport selection
        st.header("🏈 Sport Selection")
        sports = get_sports()
        sport_options = {s['title']: s['key'] for s in sports if s['active']}
        
        if sport_options:
            selected_sport_title = st.selectbox(
                "Choose Sport",
                options=list(sport_options.keys()),
                index=list(sport_options.keys()).index('NBA') if 'NBA' in sport_options else 0
            )
            selected_sport = sport_options[selected_sport_title]
        else:
            st.warning("Could not load sports")
            selected_sport = "basketball_nba"
            selected_sport_title = "NBA"
        
        st.divider()
        
        # Anchor persona
        st.header("🎭 Anchor Persona")
        selected_persona = st.selectbox(
            "Choose Your Anchor Style",
            options=list(ANCHOR_PERSONAS.keys())
        )
        
        st.divider()
        
        # Video settings
        st.header("🎥 Video Settings")
        test_mode = st.checkbox("Test Mode (no credits used)", value=True,
                                help="Generates watermarked preview")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fetch odds
        events, headers = get_odds(selected_sport)
        
        if events:
            st.subheader(f"📊 Upcoming {selected_sport_title} Games")
            
            # Game selector
            game_options = {
                f"{e['away_team']} @ {e['home_team']}": i 
                for i, e in enumerate(events[:10])
            }
            selected_game = st.selectbox("Select Game for Broadcast", options=list(game_options.keys()))
            selected_event = events[game_options[selected_game]]
            
            # Show odds summary
            with st.expander("📈 View Current Odds", expanded=True):
                odds_cols = st.columns(min(4, len(selected_event.get('bookmakers', []))))
                for i, bookmaker in enumerate(selected_event.get('bookmakers', [])[:4]):
                    with odds_cols[i]:
                        st.markdown(f"**{bookmaker['title']}**")
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'h2h':
                                for outcome in market['outcomes']:
                                    prefix = "+" if outcome['price'] > 0 else ""
                                    st.write(f"{outcome['name'][:12]}: {prefix}{outcome['price']}")
        else:
            st.warning("No games available. Try a different sport.")
            selected_event = None
    
    with col2:
        st.subheader("🎬 Broadcast Controls")
        
        if st.button("🎙️ Generate Script", type="primary", use_container_width=True):
            if not GROQ_API_KEY:
                st.error("Please set GROQ_API_KEY in your .env file")
            elif not selected_event:
                st.error("No game selected")
            else:
                with st.spinner("Gathering intel..."):
                    # Get news
                    teams = f"{selected_event['away_team']} {selected_event['home_team']}"
                    news = get_news(f"{teams} game preview injury", max_results=5)
                    
                    # Format data
                    odds_text = format_odds_for_script(selected_event)
                    news_text = format_news_for_script(news)
                    
                    # Generate script
                    script = generate_script(
                        selected_event,
                        odds_text,
                        news_text,
                        ANCHOR_PERSONAS[selected_persona]
                    )
                    
                    st.session_state['generated_script'] = script
                    st.session_state['selected_event'] = selected_event
    
    # Script display and video generation
    if 'generated_script' in st.session_state:
        st.divider()
        
        col_script, col_video = st.columns([1, 1])
        
        with col_script:
            st.subheader("📝 Anchor Script")
            
            # Editable script
            edited_script = st.text_area(
                "Edit your script before recording:",
                value=st.session_state['generated_script'],
                height=300
            )
            
            word_count = len(edited_script.split())
            estimated_duration = word_count / 150 * 60  # ~150 words per minute
            st.caption(f"📊 {word_count} words • ~{estimated_duration:.0f} seconds")
        
        with col_video:
            st.subheader("🎥 Video Generation")
            
            if HEYGEN_API_KEY:
                # Avatar selection
                with st.spinner("Loading avatars..."):
                    avatars = get_heygen_avatars(HEYGEN_API_KEY)
                    voices = get_heygen_voices(HEYGEN_API_KEY)
                
                if avatars:
                    avatar_options = {
                        f"{a.get('avatar_name', 'Unknown')} ({a.get('gender', 'N/A')})": a['avatar_id']
                        for a in avatars[:20] if a.get('avatar_id')
                    }
                    selected_avatar = st.selectbox("Choose Avatar", options=list(avatar_options.keys()))
                    avatar_id = avatar_options.get(selected_avatar)
                else:
                    st.warning("Could not load avatars")
                    avatar_id = None
                
                if voices:
                    voice_options = {
                        f"{v.get('name', 'Unknown')} - {v.get('language', 'EN')}": v['voice_id']
                        for v in voices[:30] if v.get('voice_id')
                    }
                    selected_voice = st.selectbox("Choose Voice", options=list(voice_options.keys()))
                    voice_id = voice_options.get(selected_voice)
                else:
                    st.warning("Could not load voices")
                    voice_id = None
                
                # Generate video button
                if st.button("🎬 Generate Video", type="primary", use_container_width=True):
                    if avatar_id and voice_id:
                        with st.spinner("🎥 Generating your broadcast... This takes 1-3 minutes"):
                            video_id = generate_heygen_video(
                                HEYGEN_API_KEY,
                                edited_script,
                                avatar_id,
                                voice_id,
                                test_mode=test_mode
                            )

                            if video_id:
                                st.session_state['video_id'] = video_id
                                st.success(f"Video generation started! ID: {video_id}")

                                # Poll for completion
                                progress_bar = st.progress(0)
                                status_text = st.empty()

                                for i in range(60):  # Wait up to 3 minutes
                                    time.sleep(3)
                                    status = check_video_status(HEYGEN_API_KEY, video_id)
                                    current_status = status.get('status', 'unknown')
                                    
                                    progress_bar.progress(min((i + 1) * 2, 100))
                                    status_text.text(f"Status: {current_status}")
                                    
                                    if current_status == 'completed':
                                        video_url = status.get('video_url')
                                        st.session_state['video_url'] = video_url
                                        st.success("✅ Video ready!")
                                        break
                                    elif current_status == 'failed':
                                        st.error("❌ Video generation failed")
                                        break
                    else:
                        st.error("Please select avatar and voice")
            else:
                st.info("Set HEYGEN_API_KEY in your .env file to enable video generation")
    
    # Video player
    if 'video_url' in st.session_state and st.session_state['video_url']:
        st.divider()
        st.subheader("📺 Your Broadcast")
        st.video(st.session_state['video_url'])
        st.markdown(f"[Download Video]({st.session_state['video_url']})")
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **🎓 Modern AI Pro Workshop** | Build Your Own AI News Channel
    
    *What you built that ChatGPT/Claude can't do:*
    - ✅ Real-time odds from multiple bookmakers
    - ✅ Custom news aggregation
    - ✅ Personalized anchor personas
    - ✅ AI-generated video broadcasts
    - ✅ Full source transparency
    """)

if __name__ == "__main__":
    main()