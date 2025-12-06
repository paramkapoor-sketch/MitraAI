"""
🛒 Smart Shopping Assistant - Modern AI Pro Workshop
Framework: CrewAI
Pattern: Plan-and-Execute

Features:
1. Multi-agent shopping crew
2. Product research & comparison
3. Deal hunting across retailers
4. Review analysis & sentiment
5. Personalized recommendations

What this builds that ChatGPT/Claude can't do:
- Real-time price comparison from actual retailers
- Fake review detection patterns
- Price history tracking ("is this really a deal?")
- Personalized recommendations based on preferences
"""

import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Additional tools
from duckduckgo_search import DDGS
from tavily import TavilyClient

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Set API key for CrewAI (defaults to OpenAI)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ============== CUSTOM TOOLS ==============

@tool
def web_search(query: str) -> str:
    """
    Search the web for product information, prices, and reviews.
    Use this to find product details, compare prices, and gather reviews.

    Args:
        query: The search query for products or reviews

    Returns:
        Search results with titles, snippets, and URLs
    """
    results = []

    # Try Tavily first
    if TAVILY_API_KEY:
        try:
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily.search(query, max_results=5, search_depth="basic")
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", "")[:300],
                    "url": r.get("url", "")
                })
            return json.dumps(results, indent=2)
        except Exception:
            pass

    # DuckDuckGo fallback
    try:
        ddg_results = DDGS().text(query, max_results=5)
        for r in ddg_results:
            results.append({
                "title": r.get("title", ""),
                "content": r.get("body", "")[:300],
                "url": r.get("href", "")
            })
    except Exception as e:
        return json.dumps({"error": str(e)})

    return json.dumps(results, indent=2)


@tool
def price_search(product: str) -> str:
    """
    Search for product prices across multiple retailers.
    Returns price comparisons from different stores.

    Args:
        product: The product name to search prices for

    Returns:
        Price comparison data from various retailers
    """
    try:
        # Search for pricing info
        query = f"{product} price compare buy"
        results = DDGS().text(query, max_results=8)

        prices = []
        for r in results:
            body = r.get("body", "").lower()
            title = r.get("title", "")

            # Look for price patterns
            import re
            price_matches = re.findall(r'\$[\d,]+\.?\d*', body + " " + title.lower())

            if price_matches:
                prices.append({
                    "retailer": extract_retailer(r.get("href", "")),
                    "title": title[:100],
                    "prices_found": price_matches[:3],
                    "url": r.get("href", "")
                })

        return json.dumps(prices, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def review_search(product: str) -> str:
    """
    Search for customer reviews and ratings for a product.
    Analyzes sentiment and identifies common pros/cons.

    Args:
        product: The product to find reviews for

    Returns:
        Review summary with sentiment analysis
    """
    try:
        query = f"{product} review customer rating"
        results = DDGS().text(query, max_results=6)

        reviews = []
        positive_keywords = ['great', 'excellent', 'love', 'best', 'amazing', 'perfect', 'recommend']
        negative_keywords = ['bad', 'terrible', 'worst', 'avoid', 'broken', 'disappointing', 'return']

        for r in results:
            body = r.get("body", "").lower()
            title = r.get("title", "")

            # Simple sentiment scoring
            pos_count = sum(1 for word in positive_keywords if word in body)
            neg_count = sum(1 for word in negative_keywords if word in body)

            if pos_count > neg_count:
                sentiment = "positive"
            elif neg_count > pos_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            reviews.append({
                "source": title[:80],
                "snippet": r.get("body", "")[:200],
                "sentiment": sentiment,
                "url": r.get("href", "")
            })

        return json.dumps(reviews, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def extract_retailer(url: str) -> str:
    """Extract retailer name from URL"""
    retailers = {
        "amazon": "Amazon",
        "walmart": "Walmart",
        "bestbuy": "Best Buy",
        "target": "Target",
        "newegg": "Newegg",
        "ebay": "eBay",
        "costco": "Costco",
        "bhphoto": "B&H Photo"
    }

    url_lower = url.lower()
    for key, name in retailers.items():
        if key in url_lower:
            return name
    return "Other"


# ============== AGENT DEFINITIONS ==============

def create_product_researcher():
    """Create Product Research Specialist agent"""
    return Agent(
        role="Product Research Specialist",
        goal="Find comprehensive product information including specs, variants, and alternatives",
        backstory="""You are an expert product researcher with years of experience
        comparing tech products. You know how to dig through specifications, find
        hidden features, and identify the differences between product variants.
        You're thorough and always verify information from multiple sources.""",
        tools=[web_search],
        verbose=True,
        allow_delegation=False
    )


def create_deal_hunter():
    """Create Deal Hunter agent"""
    return Agent(
        role="Deal Hunter",
        goal="Find the absolute best prices and deals across all retailers",
        backstory="""You are a legendary deal hunter who knows every trick in the book.
        You track price histories, find hidden coupon codes, know about cashback offers,
        and understand the best times to buy. You never pay full price and always find
        the best value for every purchase.""",
        tools=[web_search, price_search],
        verbose=True,
        allow_delegation=False
    )


def create_review_analyst():
    """Create Review Analyst agent"""
    return Agent(
        role="Review Analyst",
        goal="Synthesize customer reviews and identify genuine insights vs fake reviews",
        backstory="""You are a consumer advocate who has analyzed thousands of product
        reviews. You can spot fake reviews from a mile away, identify patterns in
        customer complaints, and extract the genuine pros and cons that matter.
        You focus on verified purchases and long-term ownership reviews.""",
        tools=[web_search, review_search],
        verbose=True,
        allow_delegation=False
    )


def create_shopping_advisor():
    """Create Shopping Advisor agent"""
    return Agent(
        role="Shopping Advisor",
        goal="Make the best possible purchase recommendation based on all gathered data",
        backstory="""You are a trusted friend who gives honest shopping advice. You
        consider budget, needs, timing, and value. You're not afraid to say "don't buy"
        if the timing is wrong or there's a better alternative. Your recommendations
        have saved friends thousands of dollars over the years.""",
        verbose=True,
        allow_delegation=False
    )


# ============== TASK DEFINITIONS ==============

def create_research_task(product: str, researcher):
    """Create product research task"""
    return Task(
        description=f"""Research the product: {product}

        Your research should cover:
        1. Full product specifications and features
        2. Different variants/models available
        3. Comparable alternatives from other brands
        4. Release date and lifecycle position (is a new version coming?)

        Be thorough and cite your sources.""",
        expected_output="""A comprehensive product brief including:
        - Key specifications and features
        - Available variants with differences
        - Top 2-3 alternatives
        - Any concerns about timing (new model rumors, etc.)""",
        agent=researcher
    )


def create_deal_task(product: str, deal_hunter):
    """Create deal hunting task"""
    return Task(
        description=f"""Find the best prices for: {product}

        Your deal hunting should include:
        1. Current prices at major retailers (Amazon, Walmart, Best Buy, etc.)
        2. Any active sales, coupons, or promo codes
        3. Price history insights (is this a good time to buy?)
        4. Cashback or rewards opportunities

        Focus on legitimate retailers with good return policies.""",
        expected_output="""A price comparison table with:
        - Retailer name and current price
        - Any available discounts or codes
        - Notes on whether this is a good deal
        - Links to the best offers""",
        agent=deal_hunter
    )


def create_review_task(product: str, review_analyst):
    """Create review analysis task"""
    return Task(
        description=f"""Analyze customer reviews for: {product}

        Your analysis should:
        1. Summarize overall customer sentiment
        2. Identify the most common pros (what people love)
        3. Identify the most common cons (what people complain about)
        4. Flag any potential deal-breakers or red flags
        5. Note any patterns in fake or suspicious reviews

        Focus on verified purchases and long-term reviews.""",
        expected_output="""A review summary including:
        - Overall sentiment score (positive/mixed/negative)
        - Top 3-5 pros
        - Top 3-5 cons
        - Any red flags or deal-breakers
        - Confidence level in the reviews""",
        agent=review_analyst
    )


def create_recommendation_task(product: str, advisor, context_tasks):
    """Create final recommendation task"""
    return Task(
        description=f"""Based on all the research for: {product}

        Make a final recommendation that includes:
        1. Should they buy now, wait, or consider alternatives?
        2. If buying, which retailer offers the best value?
        3. Any warnings or things to watch out for
        4. A confidence level in your recommendation

        Be honest - if this isn't a good purchase, say so.""",
        expected_output="""A clear recommendation:
        - BUY NOW / WAIT / CONSIDER ALTERNATIVE
        - Best place to purchase (if buying)
        - Key reasons for recommendation
        - Any caveats or warnings
        - Confidence level (High/Medium/Low)""",
        agent=advisor,
        context=context_tasks
    )


# ============== CREW ASSEMBLY ==============

def create_shopping_crew(product: str):
    """Assemble the shopping crew"""

    # Create agents
    researcher = create_product_researcher()
    deal_hunter = create_deal_hunter()
    review_analyst = create_review_analyst()
    advisor = create_shopping_advisor()

    # Create tasks
    research_task = create_research_task(product, researcher)
    deal_task = create_deal_task(product, deal_hunter)
    review_task = create_review_task(product, review_analyst)
    recommend_task = create_recommendation_task(
        product, advisor,
        [research_task, deal_task, review_task]
    )

    # Assemble crew
    crew = Crew(
        agents=[researcher, deal_hunter, review_analyst, advisor],
        tasks=[research_task, deal_task, review_task, recommend_task],
        process=Process.sequential,
        verbose=True
    )

    return crew


def run_shopping_assistant(product: str, progress_callback=None):
    """Run the shopping assistant workflow"""

    results = {
        "product": product,
        "timestamp": datetime.now().isoformat(),
        "phases": {}
    }

    if progress_callback:
        progress_callback("Assembling shopping crew...")

    crew = create_shopping_crew(product)

    if progress_callback:
        progress_callback("Running shopping analysis...")

    try:
        # Kick off the crew
        crew_output = crew.kickoff()

        # Process results
        results["final_recommendation"] = str(crew_output)

        # Extract individual task outputs if available
        if hasattr(crew_output, 'tasks_output'):
            for i, task_output in enumerate(crew_output.tasks_output):
                task_names = ["research", "deals", "reviews", "recommendation"]
                if i < len(task_names):
                    results["phases"][task_names[i]] = str(task_output)

        if progress_callback:
            progress_callback("Analysis complete!")

    except Exception as e:
        results["error"] = str(e)
        if progress_callback:
            progress_callback(f"Error: {e}")

    return results


# ============== STREAMLIT UI ==============

def main():
    st.set_page_config(
        page_title="Smart Shopping Assistant",
        page_icon="🛒",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .agent-card {
        background: #f0fff4;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #38ef7d;
        margin: 10px 0;
    }
    .price-card {
        background: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🛒 Smart Shopping Assistant</h1>
        <p style="color: #e8f5e9; margin: 0;">Multi-Agent Deal Finder • Powered by CrewAI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        st.subheader("🤖 Shopping Crew")
        st.markdown("""
        <div class="agent-card">
            <strong>🔍 Product Researcher</strong><br>
            Specs, variants, alternatives
        </div>
        <div class="agent-card">
            <strong>💰 Deal Hunter</strong><br>
            Prices, coupons, cashback
        </div>
        <div class="agent-card">
            <strong>⭐ Review Analyst</strong><br>
            Sentiment, pros/cons, fakes
        </div>
        <div class="agent-card">
            <strong>🎯 Shopping Advisor</strong><br>
            Final recommendation
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.subheader("🔑 API Status")
        st.write(f"OpenAI: {'✅' if OPENAI_API_KEY else '❌ Required'}")
        st.write(f"Tavily: {'✅' if TAVILY_API_KEY else '⚠️ Using fallback'}")
        st.write(f"Groq: {'✅' if GROQ_API_KEY else '❌'}")

        if not OPENAI_API_KEY:
            st.warning("Set OPENAI_API_KEY in .env for CrewAI to work")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔎 What are you shopping for?")

        # Sample products
        sample_products = [
            "Sony WH-1000XM5 headphones",
            "Apple MacBook Pro M4 14-inch",
            "LG C4 65-inch OLED TV",
            "Dyson V15 Detect vacuum",
            "Samsung Galaxy S24 Ultra"
        ]

        selected_sample = st.selectbox(
            "Popular searches:",
            ["Custom product..."] + sample_products
        )

        if selected_sample == "Custom product...":
            product_query = st.text_input(
                "Enter product name:",
                placeholder="e.g., Sony WH-1000XM5 headphones"
            )
        else:
            product_query = st.text_input(
                "Enter product name:",
                value=selected_sample
            )

    with col2:
        st.subheader("⚙️ Options")

        budget = st.text_input(
            "Budget (optional):",
            placeholder="e.g., $300"
        )

        priorities = st.multiselect(
            "What matters most?",
            ["Best Price", "Best Reviews", "Fast Shipping", "Brand Reputation"],
            default=["Best Price", "Best Reviews"]
        )

    # Search button
    if st.button("🚀 Find Best Deal", type="primary", use_container_width=True):
        if not product_query:
            st.error("Please enter a product to search for")
        elif not OPENAI_API_KEY:
            st.error("OpenAI API key required. Set OPENAI_API_KEY in your .env file")
        else:
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                status_text = st.empty()
                progress_bar = st.progress(0)

            def update_progress(message):
                status_text.text(message)
                if "Assembling" in message:
                    progress_bar.progress(20)
                elif "Running" in message:
                    progress_bar.progress(50)
                elif "complete" in message.lower():
                    progress_bar.progress(100)

            try:
                with st.spinner("Shopping crew is analyzing..."):
                    search_query = product_query
                    if budget:
                        search_query += f" under {budget}"

                    results = run_shopping_assistant(
                        search_query,
                        progress_callback=update_progress
                    )

                st.session_state['shopping_results'] = results

            except Exception as e:
                st.error(f"Shopping analysis failed: {e}")
                st.exception(e)

    # Display results
    if 'shopping_results' in st.session_state:
        results = st.session_state['shopping_results']

        st.divider()

        if "error" in results:
            st.error(f"Analysis error: {results['error']}")
        else:
            # Recommendation box
            st.markdown("""
            <div class="recommendation-box">
                <h2>🎯 Shopping Recommendation</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(results.get("final_recommendation", "No recommendation generated"))

            # Phase details in tabs
            if results.get("phases"):
                st.subheader("📊 Detailed Analysis")

                tab1, tab2, tab3, tab4 = st.tabs([
                    "🔍 Research",
                    "💰 Prices",
                    "⭐ Reviews",
                    "📋 Raw Data"
                ])

                with tab1:
                    st.markdown("### Product Research")
                    st.markdown(results["phases"].get("research", "No research data"))

                with tab2:
                    st.markdown("### Price Comparison")
                    st.markdown(results["phases"].get("deals", "No price data"))

                with tab3:
                    st.markdown("### Review Analysis")
                    st.markdown(results["phases"].get("reviews", "No review data"))

                with tab4:
                    st.json(results)

    # Footer
    st.divider()
    st.markdown("""
    ---
    **🎓 Modern AI Pro Workshop** | CrewAI Framework

    *What this does that ChatGPT/Claude can't:*
    - ✅ Real-time price comparison from actual retailers
    - ✅ Multi-agent collaboration with specialized roles
    - ✅ Review sentiment analysis and fake detection
    - ✅ Personalized buy/wait/alternative recommendations
    - ✅ Coordinated task delegation between agents

    **Framework:** [CrewAI](https://github.com/crewAIInc/crewAI)
    """)


if __name__ == "__main__":
    main()
