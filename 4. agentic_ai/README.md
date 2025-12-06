# 🎬 AI Sports News Channel

**Modern AI Pro Workshop: Build Your Own ESPN**

## The Big Idea

Students build a **personalized AI sports news channel** that does what ChatGPT/Claude fundamentally cannot:

```
Real-time Odds → News Aggregation → LLM Analysis → AI Avatar Video
```

## Two Apps Included

### 1. `sports_intel_agent.py` - Foundation App
Basic sports intelligence with odds comparison and value detection.

### 2. `sports_news_channel.py` - Full News Channel 🌟
Complete broadcast system with HeyGen video avatar generation.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic app
streamlit run sports_intel_agent.py

# Run full news channel
streamlit run sports_news_channel.py
```

## API Keys Needed

| API | Cost | Get It |
|-----|------|--------|
| **Groq** | Free | [console.groq.com](https://console.groq.com) |
| **The Odds API** | Included | Already in code (500/month) |
| **HeyGen** | Free 10 credits | [heygen.com](https://heygen.com) |

---

## What Students Build (That AI Chatbots Can't Do)

| Feature | ChatGPT/Claude | Your App |
|---------|---------------|----------|
| Real-time multi-bookmaker odds | ❌ | ✅ |
| Value bet / arbitrage detection | ❌ | ✅ |
| Custom saveable analyst personas | ❌ | ✅ |
| See the raw source data | ❌ | ✅ |
| Generate AI avatar video | ❌ | ✅ |
| Personalized sports broadcast | ❌ | ✅ |

---

## App Features

### 📊 Live Odds Tab
- Real-time odds from FanDuel, DraftKings, BetMGM, Caesars
- American odds format
- Multiple sports (NFL, NBA, MLB, Soccer, Cricket, etc.)

### 💰 Value Finder Tab
- Automated arbitrage opportunity detection
- Best/worst price comparison across books
- Odds spread calculation

### 📰 News + Analysis Tab
- DuckDuckGo news aggregation
- 5 built-in anchor personas:
  - 🎙️ Classic ESPN Anchor
  - 📊 Stats Analyst
  - 🔥 Hot Take Host
  - 🏏 Cricket Commentator
  - 🎯 Betting Sharp
- Or create your own!

### 🎬 Video Generation (HeyGen)
- Choose from 100+ AI avatars
- Multiple voice options
- Professional broadcast output
- Test mode (no credits) for demos

---

## Workshop Extensions

**30-Minute Builds:**
1. Add more analyst personas (Navjot Sidhu, Skip Bayless)
2. Multi-regional news comparison (US vs India coverage)
3. Historical odds tracking

**60-Minute Builds:**
4. Automated daily briefing emails
5. Discord/Slack bot integration
6. Prediction tracking & accuracy scoring

**Advanced:**
7. Build a "Research Session" that tracks topics over time
8. Add sentiment analysis on news sources
9. Create video playlist for multiple games

---

## File Structure

```
├── sports_intel_agent.py    # Basic app (no video)
├── sports_news_channel.py   # Full app with HeyGen
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## Credits & Costs

### The Odds API (Included)
- 500 requests/month free
- Sports endpoint = FREE
- Each odds query = 1 credit

### HeyGen
- 10 credits/month free
- 1 credit ≈ 1 minute of video
- Test mode = unlimited (watermarked)

### Groq
- Generous free tier
- llama-3.3-70b-versatile model

---

## Workshop Talking Points

1. **Why this matters**: AI agents that DO things, not just chat
2. **API composition**: Combining multiple data sources
3. **Personalization**: Custom personas are a moat
4. **Transparency**: Users can see the raw data (trust)
5. **Multimodal output**: Text → Video is the future

---

*Built for Modern AI Pro Agentic AI Workshop*
*"Monday-ready" tools students actually use*