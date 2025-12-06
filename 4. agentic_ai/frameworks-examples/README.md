# Agentic AI Workshop: Multi-Framework Examples
## Modern AI Pro | 3 Orchestration Frameworks × 3 Use Cases

---

## Framework-to-Use Case Mapping

Based on the **Pattern Comparison** and **Agent Architecture** (Profile, Memory, Planning, Action):

| Framework | Use Case | Pattern | Why This Fit |
|-----------|----------|---------|--------------|
| **Strands Agents** | Second Brain Research | Orchestrator-Workers | Multi-agent workflow, specialized roles, AWS-native |
| **CrewAI** | Shopping Assistant | Plan-and-Execute | Role-based crews, task delegation, sequential processing |
| **LangGraph** | Stock Analysis (Polygon) | ReAct | Stateful graphs, tool use, real-time data, conditional routing |

---

## Example 1: Second Brain Research Assistant (Strands Agents)

### Why Strands?
- AWS-backed, production-ready
- Clean multi-agent workflow orchestration
- Built-in suppression of intermediate outputs
- MCP (Model Context Protocol) for tool integration

### Architecture
```
User Query → Router Agent → [Researcher | Analyst | Synthesizer] → Final Report
                              ↓           ↓            ↓
                          Web Search   Verify      Compile
                          News API     Cross-ref   Format
```

### Agent Roles

| Agent | Role | Tools | Output |
|-------|------|-------|--------|
| **Topic Router** | Determines research type | None | Route decision |
| **Web Researcher** | Gathers information | http_request, news_api | Raw findings |
| **Fact Checker** | Verifies claims | search, cross-reference | Verified facts |
| **Synthesizer** | Compiles final report | None | Structured report |

### Key Code Structure
```python
from strands import Agent

# Researcher with web capabilities
researcher = Agent(
    system_prompt="""You are a Research Agent. 
    1. Search for authoritative sources on the topic
    2. Extract key facts with citations
    3. Note conflicting information""",
    tools=[http_request, news_search],
    callback_handler=None  # Suppress intermediate output
)

# Fact Checker
fact_checker = Agent(
    system_prompt="""You are a Fact Verification Agent.
    1. Cross-reference claims across sources
    2. Rate confidence (1-5) for each fact
    3. Flag any contradictions""",
    callback_handler=None
)

# Synthesizer
synthesizer = Agent(
    system_prompt="""You are a Report Synthesis Agent.
    1. Compile verified facts into coherent narrative
    2. Structure with clear sections
    3. Include confidence levels and citations"""
)

def research_workflow(query):
    # Step 1: Research
    findings = researcher(f"Research: {query}")
    
    # Step 2: Verify
    verified = fact_checker(f"Verify these findings:\n{findings}")
    
    # Step 3: Synthesize
    report = synthesizer(f"Create report from:\n{verified}")
    
    return report
```

### Differentiator from ChatGPT/Claude
- **Persistent research sessions** - tracks what's been covered
- **Source verification** - cross-references automatically
- **Structured memory** - remembers research history
- **Contradiction detection** - flags conflicting sources

### Workshop Build Time: 45 minutes

---

## Example 2: Smart Shopping Assistant (CrewAI)

### Why CrewAI?
- Role-based agent architecture
- Task-centric execution
- Natural delegation between agents
- Great for e-commerce workflows

### Architecture
```
User: "Find best deal on Sony WH-1000XM5"
            ↓
┌──────────────────────────────────────────┐
│           Shopping Crew                   │
├──────────────────────────────────────────┤
│  Product Researcher → Deal Hunter →      │
│  Review Analyst → Recommendation Agent   │
└──────────────────────────────────────────┘
            ↓
Final: "Best price at Amazon $278, 
        verified 4.7★, price dropped 15% this week"
```

### Crew Roles

| Agent | Goal | Backstory | Tools |
|-------|------|-----------|-------|
| **Product Researcher** | Find product specs & variants | Expert in product categorization | web_search |
| **Deal Hunter** | Find lowest prices across retailers | Knows all the discount tricks | price_search, deal_api |
| **Review Analyst** | Summarize reviews & sentiment | Consumer advocate | review_scraper |
| **Advisor** | Make final recommendation | Trusted shopping consultant | None |

### Key Code Structure
```python
from crewai import Agent, Task, Crew, Process

# Define Agents
product_researcher = Agent(
    role="Product Research Specialist",
    goal="Find comprehensive product information and variants",
    backstory="You're an expert at finding product specifications, 
               comparing models, and identifying key features.",
    tools=[web_search_tool],
    verbose=True
)

deal_hunter = Agent(
    role="Deal Hunter",
    goal="Find the absolute best prices across all retailers",
    backstory="You know every trick to find deals - price history,
               coupon codes, cashback offers, and timing.",
    tools=[price_comparison_tool],
    verbose=True
)

review_analyst = Agent(
    role="Review Analyst", 
    goal="Synthesize customer sentiment and identify real issues",
    backstory="You can spot fake reviews and extract genuine insights
               from thousands of customer opinions.",
    tools=[review_search_tool],
    verbose=True
)

advisor = Agent(
    role="Shopping Advisor",
    goal="Make the best recommendation based on all gathered data",
    backstory="You're a trusted friend who gives honest advice,
               considering budget, needs, and timing.",
    verbose=True
)

# Define Tasks
research_task = Task(
    description="Research {product} - specs, variants, alternatives",
    expected_output="Comprehensive product brief with key specs",
    agent=product_researcher
)

deal_task = Task(
    description="Find best prices for {product} across retailers",
    expected_output="Price comparison table with links and notes",
    agent=deal_hunter
)

review_task = Task(
    description="Analyze reviews for {product}",
    expected_output="Sentiment summary with pros/cons and red flags",
    agent=review_analyst
)

recommend_task = Task(
    description="Based on research, make final recommendation",
    expected_output="Clear buy/wait/alternative recommendation with reasoning",
    agent=advisor
)

# Assemble Crew
shopping_crew = Crew(
    agents=[product_researcher, deal_hunter, review_analyst, advisor],
    tasks=[research_task, deal_task, review_task, recommend_task],
    process=Process.sequential,
    verbose=True
)

# Run
result = shopping_crew.kickoff(inputs={"product": "Sony WH-1000XM5"})
```

### Differentiator from ChatGPT/Claude
- **Real-time price comparison** - actual retailer data
- **Fake review detection** - pattern analysis
- **Price history tracking** - "is this really a deal?"
- **Personalized recommendations** - learns preferences

### Workshop Build Time: 45 minutes

---

## Example 3: Stock Analysis Agent (LangGraph + Polygon API)

### Why LangGraph?
- Stateful graph-based workflows
- Conditional branching (different paths for different analyses)
- Tool integration with fallbacks
- Perfect for financial data pipelines

### Architecture
```
User: "Analyze NVDA for potential buy"
            ↓
    ┌───────────────┐
    │ Router Node   │
    └───────┬───────┘
            ↓
    ┌───────────────────────────────────────┐
    │        Parallel Analysis Nodes         │
    ├───────────┬───────────┬───────────────┤
    │ Technical │ Fundamental│    News      │
    │ Analysis  │  Analysis  │  Sentiment   │
    └─────┬─────┴─────┬─────┴───────┬───────┘
          └───────────┴─────────────┘
                      ↓
            ┌─────────────────┐
            │ Synthesis Node  │
            └────────┬────────┘
                     ↓
            ┌─────────────────┐
            │ Recommendation  │
            │ (Buy/Sell/Hold) │
            └─────────────────┘
```

### Node Functions

| Node | Purpose | Tools | Output |
|------|---------|-------|--------|
| **Data Fetcher** | Get stock data | Polygon API | OHLCV, fundamentals |
| **Technical Analyst** | Calculate indicators | ta library | RSI, MACD, signals |
| **Fundamental Analyst** | Evaluate financials | Polygon API | P/E, margins, growth |
| **News Analyst** | Sentiment analysis | News API + LLM | Sentiment score |
| **Synthesizer** | Combine analyses | None | Unified view |
| **Recommender** | Final call | None | Buy/Sell/Hold + confidence |

### Key Code Structure
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import requests

# State definition
class StockAnalysisState(TypedDict):
    ticker: str
    price_data: dict
    technical_signals: dict
    fundamental_data: dict
    news_sentiment: dict
    synthesis: str
    recommendation: str
    confidence: float

# Polygon API Tool
POLYGON_API_KEY = "your_key_here"

def fetch_stock_data(state: StockAnalysisState) -> StockAnalysisState:
    ticker = state["ticker"]
    
    # Get price data from Polygon
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/2024-12-01"
    response = requests.get(url, params={"apiKey": POLYGON_API_KEY})
    price_data = response.json()
    
    # Get company details
    details_url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    details = requests.get(details_url, params={"apiKey": POLYGON_API_KEY}).json()
    
    state["price_data"] = price_data
    state["fundamental_data"] = details
    return state

def technical_analysis(state: StockAnalysisState) -> StockAnalysisState:
    """Calculate RSI, MACD, Bollinger Bands"""
    import pandas as pd
    import ta
    
    prices = state["price_data"]["results"]
    df = pd.DataFrame(prices)
    
    # Calculate indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['c']).rsi()
    df['macd'] = ta.trend.MACD(df['c']).macd()
    
    # Generate signals
    latest_rsi = df['rsi'].iloc[-1]
    signal = "oversold" if latest_rsi < 30 else "overbought" if latest_rsi > 70 else "neutral"
    
    state["technical_signals"] = {
        "rsi": latest_rsi,
        "macd": df['macd'].iloc[-1],
        "signal": signal
    }
    return state

def news_sentiment(state: StockAnalysisState) -> StockAnalysisState:
    """Analyze recent news sentiment"""
    from ddgs import DDGS
    
    ticker = state["ticker"]
    news = DDGS().news(f"{ticker} stock news", max_results=5)
    
    # Use LLM for sentiment
    news_text = "\n".join([n['title'] + ": " + n.get('body', '')[:200] for n in news])
    
    # Simplified sentiment (in production, use LLM)
    positive_words = ['surge', 'beat', 'growth', 'profit', 'upgrade']
    negative_words = ['fall', 'miss', 'decline', 'downgrade', 'concern']
    
    score = sum(1 for w in positive_words if w in news_text.lower())
    score -= sum(1 for w in negative_words if w in news_text.lower())
    
    state["news_sentiment"] = {
        "score": score,
        "recent_headlines": [n['title'] for n in news[:3]]
    }
    return state

def synthesize(state: StockAnalysisState) -> StockAnalysisState:
    """Combine all analyses"""
    synthesis = f"""
    STOCK ANALYSIS: {state['ticker']}
    
    Technical: RSI={state['technical_signals']['rsi']:.1f} ({state['technical_signals']['signal']})
    News Sentiment: {state['news_sentiment']['score']} (positive if >0)
    
    Key Headlines:
    {chr(10).join('- ' + h for h in state['news_sentiment']['recent_headlines'])}
    """
    state["synthesis"] = synthesis
    return state

def recommend(state: StockAnalysisState) -> StockAnalysisState:
    """Generate final recommendation"""
    rsi = state["technical_signals"]["rsi"]
    sentiment = state["news_sentiment"]["score"]
    
    # Simple scoring logic
    score = 0
    if rsi < 40: score += 2  # Potentially undervalued
    if rsi > 70: score -= 2  # Overbought
    score += sentiment
    
    if score >= 2:
        rec, conf = "BUY", 0.7
    elif score <= -2:
        rec, conf = "SELL", 0.7
    else:
        rec, conf = "HOLD", 0.5
    
    state["recommendation"] = rec
    state["confidence"] = conf
    return state

# Build Graph
workflow = StateGraph(StockAnalysisState)

# Add nodes
workflow.add_node("fetch_data", fetch_stock_data)
workflow.add_node("technical", technical_analysis)
workflow.add_node("sentiment", news_sentiment)
workflow.add_node("synthesize", synthesize)
workflow.add_node("recommend", recommend)

# Add edges
workflow.set_entry_point("fetch_data")
workflow.add_edge("fetch_data", "technical")
workflow.add_edge("fetch_data", "sentiment")  # Parallel
workflow.add_edge("technical", "synthesize")
workflow.add_edge("sentiment", "synthesize")
workflow.add_edge("synthesize", "recommend")
workflow.add_edge("recommend", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"ticker": "NVDA"})
print(result["synthesis"])
print(f"Recommendation: {result['recommendation']} (Confidence: {result['confidence']})")
```

### Polygon API Endpoints Used

| Endpoint | Purpose | Free Tier |
|----------|---------|-----------|
| `/v2/aggs/ticker/{ticker}/range/...` | Historical prices | ✅ 5 calls/min |
| `/v3/reference/tickers/{ticker}` | Company details | ✅ |
| `/v2/reference/news` | News articles | ✅ |
| `/v3/reference/tickers/{ticker}/financials` | Financials | ✅ |

### Differentiator from ChatGPT/Claude
- **Real-time market data** - actual Polygon API
- **Technical indicators** - calculated, not hallucinated
- **Stateful analysis** - tracks portfolio over time
- **Conditional workflows** - different paths for different stocks

### Workshop Build Time: 60 minutes

---

## Workshop Schedule Suggestion

### Day 1: Understanding Patterns (Theory + Demo)
| Time | Activity |
|------|----------|
| 30 min | Pattern Comparison deep-dive (ReAct, Plan-Execute, etc.) |
| 30 min | Agent Architecture (Profile, Memory, Planning, Action) |
| 45 min | Demo: Strands Second Brain |
| 45 min | Demo: CrewAI Shopping Assistant |

### Day 2: Build & Deploy
| Time | Activity |
|------|----------|
| 60 min | Build: LangGraph Stock Analyzer with Polygon |
| 30 min | Integration: Combine with Sports News Channel |
| 30 min | Deployment: Streamlit Cloud / Codespaces |
| 30 min | Extensions & Q&A |

---

## API Keys Needed

| API | Cost | Get It |
|-----|------|--------|
| **Groq** | Free | console.groq.com |
| **Polygon.io** | Free (5/min) | polygon.io |
| **DuckDuckGo** | Free | No key needed |
| **OpenAI** (optional) | Pay | platform.openai.com |

---

## Files to Create

```
workshop/
├── strands_second_brain/
│   ├── main.py
│   ├── agents.py
│   └── tools.py
├── crewai_shopping/
│   ├── main.py
│   ├── agents.py
│   ├── tasks.py
│   └── crew.py
├── langgraph_stocks/
│   ├── main.py
│   ├── nodes.py
│   ├── tools.py
│   └── graph.py
└── requirements.txt
```

---

## Key Teaching Points

### From Your Pattern Comparison Image:

| Pattern | Best For | Trade-off | Example in Workshop |
|---------|----------|-----------|---------------------|
| **ReAct** | Tool use, real-time data | Multiple LLM calls | Stock Analyzer |
| **Plan-and-Execute** | Multi-step projects | Less adaptive | Shopping Assistant |
| **Orchestrator-Workers** | Multi-domain tasks | Coordination complexity | Second Brain |

### From Your Agent Architecture Image:

| Component | Strands | CrewAI | LangGraph |
|-----------|---------|--------|-----------|
| **Profile** | System prompt per agent | Role + Goal + Backstory | Node-level prompts |
| **Memory** | Session management | Shared context | State dictionary |
| **Planning** | Workflow function | Process (sequential/hierarchical) | Graph edges |
| **Action** | Tools via MCP | Tools via LangChain | Tool nodes |

---

*Prepared for Modern AI Pro Agentic AI Workshop*
*"Monday-ready" agents students actually deploy*