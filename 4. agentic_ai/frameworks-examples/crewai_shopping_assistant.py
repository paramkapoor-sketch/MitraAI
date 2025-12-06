"""
🛒 Smart Shopping Assistant - Modern AI Pro Workshop
Framework: CrewAI
Pattern: Plan-and-Execute (Role-based Crews)

Features:
1. Product research & specification lookup
2. Multi-retailer price comparison
3. Review analysis & sentiment detection
4. Personalized recommendations
5. Deal timing intelligence

What this builds that ChatGPT/Claude can't do:
- Real-time price comparison across retailers
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

# External tools
from duckduckgo_search import DDGS
import requests

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Set API key for CrewAI (uses OpenAI by default, can use Groq)
if GROQ_API_KEY and not OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# ============== CUSTOM TOOLS ==============

@tool("Product Search Tool")
def product_search(query: str) -> str:
    """
    Search for product information, specifications, and variants.
    Use this to find detailed product specs and alternatives.

    Args:
        query: Product name or search query

    Returns:
        JSON string with product information and specifications
    """
    try:
        results = DDGS().text(
            f"{query} specifications features review",
            region='wt-wt',
            max_results=5
        )
        products = []
        for r in results:
            products.append({
                "title": r.get("title", ""),
                "description": r.get("body", ""),
                "url": r.get("href", "")
            })
        return json.dumps(products, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Price Comparison Tool")
def price_search(product: str) -> str:
    """
    Search for prices across multiple retailers.
    Finds current prices, deals, and discounts.

    Args:
        product: The exact product name to search prices for

    Returns:
        JSON string with price comparisons from different retailers
    """
    try:
        # Search for prices
        results = DDGS().text(
            f"{product} price buy USD",
            region='us-en',
            max_results=8
        )

        prices = []
        retailers = ["amazon", "bestbuy", "walmart", "target", "ebay", "newegg", "bhphoto", "adorama"]

        for r in results:
            url = r.get("href", "").lower()
            title = r.get("title", "")
            body = r.get("body", "")

            # Try to identify retailer
            retailer = "Other"
            for ret in retailers:
                if ret in url:
                    retailer = ret.capitalize()
                    break

            # Extract price if present (simple pattern matching)
            import re
            price_match = re.search(r'\$[\d,]+\.?\d*', title + " " + body)
            price = price_match.group(0) if price_match else "Check site"

            prices.append({
                "retailer": retailer,
                "title": title[:100],
                "price": price,
                "url": r.get("href", "")
            })

        return json.dumps(prices, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Review Analyzer Tool")
def review_search(product: str) -> str:
    """
    Search for and analyze customer reviews.
    Identifies sentiment, common pros/cons, and potential red flags.

    Args:
        product: Product name to search reviews for

    Returns:
        JSON string with review summaries and sentiment analysis
    """
    try:
        results = DDGS().text(
            f"{product} review pros cons rating",
            region='wt-wt',
            max_results=6
        )

        reviews = []
        for r in results:
            body = r.get("body", "").lower()

            # Simple sentiment indicators
            positive_words = ['excellent', 'great', 'amazing', 'best', 'love', 'perfect', 'recommend', 'fantastic']
            negative_words = ['terrible', 'worst', 'avoid', 'broken', 'disappointed', 'waste', 'poor', 'bad']

            pos_count = sum(1 for word in positive_words if word in body)
            neg_count = sum(1 for word in negative_words if word in body)

            if pos_count > neg_count:
                sentiment = "positive"
            elif neg_count > pos_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            reviews.append({
                "source": r.get("title", "")[:80],
                "snippet": r.get("body", "")[:300],
                "sentiment": sentiment,
                "url": r.get("href", "")
            })

        return json.dumps(reviews, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("Deal History Tool")
def deal_search(product: str) -> str:
    """
    Search for deal history and price trends.
    Helps determine if current price is actually a good deal.

    Args:
        product: Product name to check deal history

    Returns:
        JSON string with deal history and price trend information
    """
    try:
        results = DDGS().text(
            f"{product} price drop deal history lowest price",
            region='us-en',
            max_results=5
        )

        deals = []
        for r in results:
            deals.append({
                "title": r.get("title", ""),
                "info": r.get("body", "")[:300],
                "url": r.get("href", "")
            })

        return json.dumps(deals, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============== AGENT DEFINITIONS ==============

def create_product_researcher():
    """Create the Product Research Specialist agent"""
    return Agent(
        role="Product Research Specialist",
        goal="Find comprehensive product information, specifications, and viable alternatives",
        backstory="""You are an expert product researcher with years of experience
        analyzing consumer electronics, appliances, and tech products. You know how to
        find detailed specifications, identify key differentiating features, and spot
        the differences between product variants and generations. You always look for
        the full picture before making recommendations.""",
        tools=[product_search],
        verbose=True,
        allow_delegation=False,
        max_iter=10
    )


def create_deal_hunter():
    """Create the Deal Hunter agent"""
    return Agent(
        role="Deal Hunter",
        goal="Find the absolute best prices and identify genuine deals vs fake discounts",
        backstory="""You are a legendary bargain hunter who knows every trick in the book.
        You understand retailer pricing strategies, know when sales are real vs inflated
        MSRPs, track price histories, and always find coupon codes and cashback offers.
        You never let a buyer pay more than necessary and you can spot a fake deal
        from a mile away.""",
        tools=[price_search, deal_search],
        verbose=True,
        allow_delegation=False,
        max_iter=10
    )


def create_review_analyst():
    """Create the Review Analyst agent"""
    return Agent(
        role="Consumer Review Analyst",
        goal="Synthesize customer opinions and identify real product issues vs outliers",
        backstory="""You are a consumer advocate who has analyzed millions of product reviews.
        You can spot fake reviews, identify common quality issues, and separate valid
        concerns from user error. You weight professional reviews against user experiences
        and always look for patterns in complaints that indicate real problems.""",
        tools=[review_search],
        verbose=True,
        allow_delegation=False,
        max_iter=10
    )


def create_shopping_advisor():
    """Create the Shopping Advisor agent"""
    return Agent(
        role="Personal Shopping Advisor",
        goal="Make the best recommendation considering price, quality, timing, and user needs",
        backstory="""You are a trusted shopping consultant who gives honest, unbiased advice.
        You consider the buyer's budget, actual needs (not just wants), and timing.
        You're not afraid to say "wait for a better deal" or "this isn't worth it."
        Your recommendations are always backed by data and you explain your reasoning clearly.""",
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )


# ============== TASK DEFINITIONS ==============

def create_research_task(product: str, researcher: Agent):
    """Create the product research task"""
    return Task(
        description=f"""Research the product: {product}

        Your research should cover:
        1. Full product specifications and key features
        2. Different variants/models available (if any)
        3. Notable alternatives in the same category
        4. Release date and product lifecycle status
        5. Any known issues or recalls

        Be thorough but focused on information relevant to a purchase decision.""",
        expected_output="""A comprehensive product brief containing:
        - Key specifications
        - Feature highlights
        - Available variants with differences
        - Top 2-3 alternatives worth considering
        - Product age/lifecycle status""",
        agent=researcher
    )


def create_price_task(product: str, deal_hunter: Agent):
    """Create the price comparison task"""
    return Task(
        description=f"""Find the best prices for: {product}

        Your price research should include:
        1. Current prices at major retailers (Amazon, Best Buy, Walmart, etc.)
        2. Any active coupon codes or promotions
        3. Price history - is current price high, low, or average?
        4. Upcoming sales events that might offer better prices
        5. Refurbished/open-box options if significantly cheaper

        Focus on legitimate retailers with good return policies.""",
        expected_output="""A price comparison report containing:
        - Price table with retailer, price, and link
        - Best current deal identified
        - Price trend analysis (is this a good time to buy?)
        - Any applicable coupons or cashback
        - Recommendation on whether to buy now or wait""",
        agent=deal_hunter
    )


def create_review_task(product: str, analyst: Agent):
    """Create the review analysis task"""
    return Task(
        description=f"""Analyze customer reviews for: {product}

        Your analysis should cover:
        1. Overall sentiment across multiple review sources
        2. Most common praise points
        3. Most common complaints or issues
        4. Professional reviewer consensus
        5. Red flags or deal-breakers identified

        Look for patterns, not just individual opinions.""",
        expected_output="""A review analysis containing:
        - Overall sentiment score (positive/mixed/negative)
        - Top 3 praised features
        - Top 3 common complaints
        - Professional vs user review alignment
        - Any red flags or concerns
        - Reliability assessment""",
        agent=analyst
    )


def create_recommendation_task(product: str, context: str, advisor: Agent):
    """Create the final recommendation task"""
    return Task(
        description=f"""Based on all the research gathered, make a final recommendation for: {product}

        Additional context from user: {context}

        Consider:
        1. Is this product worth buying at the current price?
        2. Should they wait for a better deal?
        3. Are there better alternatives to consider?
        4. Any specific warnings or tips for the buyer?

        Be honest and direct. If it's not a good buy, say so.""",
        expected_output="""A clear recommendation containing:
        - BUY NOW / WAIT / CONSIDER ALTERNATIVE verdict
        - Confidence level (high/medium/low)
        - Best place to buy and at what price
        - Key reasons for the recommendation
        - Any important caveats or warnings
        - Alternative suggestion if applicable""",
        agent=advisor
    )


# ============== CREW EXECUTION ==============

def run_shopping_crew(product: str, user_context: str = "", progress_callback=None):
    """Execute the shopping assistant crew workflow"""

    results = {
        "product": product,
        "timestamp": datetime.now().isoformat(),
        "phases": {}
    }

    # Create agents
    researcher = create_product_researcher()
    deal_hunter = create_deal_hunter()
    review_analyst = create_review_analyst()
    advisor = create_shopping_advisor()

    # Create tasks
    research_task = create_research_task(product, researcher)
    price_task = create_price_task(product, deal_hunter)
    review_task = create_review_task(product, review_analyst)

    # First crew: Research in parallel-ish (sequential but fast)
    if progress_callback:
        progress_callback("Phase 1/4: Researching product specifications...")

    research_crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )

    try:
        research_result = research_crew.kickoff()
        results["phases"]["research"] = str(research_result)
    except Exception as e:
        results["phases"]["research"] = f"Research error: {e}"

    # Price hunting
    if progress_callback:
        progress_callback("Phase 2/4: Hunting for best prices...")

    price_crew = Crew(
        agents=[deal_hunter],
        tasks=[price_task],
        process=Process.sequential,
        verbose=True
    )

    try:
        price_result = price_crew.kickoff()
        results["phases"]["pricing"] = str(price_result)
    except Exception as e:
        results["phases"]["pricing"] = f"Pricing error: {e}"

    # Review analysis
    if progress_callback:
        progress_callback("Phase 3/4: Analyzing customer reviews...")

    review_crew = Crew(
        agents=[review_analyst],
        tasks=[review_task],
        process=Process.sequential,
        verbose=True
    )

    try:
        review_result = review_crew.kickoff()
        results["phases"]["reviews"] = str(review_result)
    except Exception as e:
        results["phases"]["reviews"] = f"Review error: {e}"

    # Final recommendation
    if progress_callback:
        progress_callback("Phase 4/4: Generating recommendation...")

    # Build context from previous phases
    full_context = f"""
    User's requirements: {user_context}

    PRODUCT RESEARCH:
    {results["phases"].get("research", "Not available")}

    PRICE ANALYSIS:
    {results["phases"].get("pricing", "Not available")}

    REVIEW ANALYSIS:
    {results["phases"].get("reviews", "Not available")}
    """

    recommendation_task = create_recommendation_task(product, full_context, advisor)

    recommendation_crew = Crew(
        agents=[advisor],
        tasks=[recommendation_task],
        process=Process.sequential,
        verbose=True
    )

    try:
        recommendation_result = recommendation_crew.kickoff()
        results["phases"]["recommendation"] = str(recommendation_result)
        results["final_recommendation"] = str(recommendation_result)
    except Exception as e:
        results["phases"]["recommendation"] = f"Recommendation error: {e}"
        results["final_recommendation"] = f"Could not generate recommendation: {e}"

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
    .crew-member {
        background: #f0fff4;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #38ef7d;
        margin: 10px 0;
    }
    .verdict-buy {
        background: #d4edda;
        border: 2px solid #28a745;
        padding: 20px;
        border-radius: 10px;
    }
    .verdict-wait {
        background: #fff3cd;
        border: 2px solid #ffc107;
        padding: 20px;
        border-radius: 10px;
    }
    .verdict-skip {
        background: #f8d7da;
        border: 2px solid #dc3545;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🛒 Smart Shopping Assistant</h1>
        <p style="color: #e0e0e0; margin: 0;">AI Shopping Crew • Powered by CrewAI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🤖 Shopping Crew")

        st.markdown("""
        <div class="crew-member">
            <strong>🔍 Product Researcher</strong><br>
            Specs, features, alternatives
        </div>
        <div class="crew-member">
            <strong>💰 Deal Hunter</strong><br>
            Prices, coupons, timing
        </div>
        <div class="crew-member">
            <strong>⭐ Review Analyst</strong><br>
            Sentiment, issues, reliability
        </div>
        <div class="crew-member">
            <strong>🎯 Shopping Advisor</strong><br>
            Final recommendation
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.subheader("🔑 API Status")
        st.write(f"OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")
        st.write(f"Groq: {'✅ (fallback)' if GROQ_API_KEY else '❌'}")
        st.write(f"Tavily: {'✅' if TAVILY_API_KEY else '❌'}")

        st.divider()

        st.subheader("💡 Tips")
        st.markdown("""
        - Be specific with product names
        - Include model numbers if known
        - Mention your budget range
        - Tell us what's important to you
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔎 What are you shopping for?")

        # Sample products
        sample_products = [
            "Sony WH-1000XM5 headphones",
            "Apple MacBook Air M3",
            "Samsung Galaxy S24 Ultra",
            "LG C4 65-inch OLED TV",
            "Dyson V15 Detect vacuum"
        ]

        selected_sample = st.selectbox(
            "Try a sample product:",
            ["Enter custom product..."] + sample_products
        )

        if selected_sample == "Enter custom product...":
            product_query = st.text_input(
                "Product name:",
                placeholder="e.g., Sony WH-1000XM5 wireless headphones"
            )
        else:
            product_query = st.text_input(
                "Product name:",
                value=selected_sample
            )

    with col2:
        st.subheader("📋 Your Preferences")

        budget = st.text_input(
            "Budget (optional):",
            placeholder="e.g., Under $300"
        )

        priorities = st.multiselect(
            "What matters most?",
            ["Best price", "Highest quality", "Fastest shipping",
             "Best warranty", "Most features", "Best reviews"],
            default=["Best price", "Best reviews"]
        )

        timing = st.radio(
            "When do you need it?",
            ["ASAP", "Within a week", "Can wait for a deal"],
            index=2
        )

    # Build context
    user_context = f"Budget: {budget}. Priorities: {', '.join(priorities)}. Timing: {timing}"

    # Search button
    if st.button("🚀 Find Best Deal", type="primary", use_container_width=True):
        if not product_query:
            st.error("Please enter a product name")
        elif not OPENAI_API_KEY and not GROQ_API_KEY:
            st.error("Please set OPENAI_API_KEY or GROQ_API_KEY in your .env file")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(message):
                if "1/4" in message:
                    progress_bar.progress(25)
                elif "2/4" in message:
                    progress_bar.progress(50)
                elif "3/4" in message:
                    progress_bar.progress(75)
                elif "4/4" in message:
                    progress_bar.progress(90)
                status_text.text(message)

            try:
                with st.spinner("Shopping crew is working..."):
                    results = run_shopping_crew(
                        product_query,
                        user_context,
                        progress_callback=update_progress
                    )

                progress_bar.progress(100)
                status_text.text("Analysis complete!")

                st.session_state['shopping_results'] = results

            except Exception as e:
                st.error(f"Shopping crew encountered an error: {e}")
                st.exception(e)

    # Display results
    if 'shopping_results' in st.session_state:
        results = st.session_state['shopping_results']

        st.divider()

        # Final recommendation highlight
        st.subheader("🎯 Recommendation")

        recommendation = results.get("final_recommendation", "")

        # Determine verdict styling
        rec_lower = recommendation.lower()
        if "buy now" in rec_lower or "recommend" in rec_lower:
            verdict_class = "verdict-buy"
            verdict_icon = "✅"
        elif "wait" in rec_lower or "hold" in rec_lower:
            verdict_class = "verdict-wait"
            verdict_icon = "⏳"
        else:
            verdict_class = "verdict-skip"
            verdict_icon = "⚠️"

        st.markdown(f"""
        <div class="{verdict_class}">
            <h3>{verdict_icon} Shopping Advisor's Verdict</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(recommendation)

        # Detailed tabs
        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs([
            "📦 Product Research",
            "💰 Price Analysis",
            "⭐ Review Analysis",
            "📊 Raw Data"
        ])

        with tab1:
            st.markdown("## Product Research")
            st.markdown(results.get("phases", {}).get("research", "No research available"))

        with tab2:
            st.markdown("## Price Comparison")
            st.markdown(results.get("phases", {}).get("pricing", "No pricing available"))

        with tab3:
            st.markdown("## Review Analysis")
            st.markdown(results.get("phases", {}).get("reviews", "No reviews available"))

        with tab4:
            st.json(results)

    # Footer
    st.divider()
    st.markdown("""
    ---
    **🎓 Modern AI Pro Workshop** | CrewAI Framework

    *What this does that ChatGPT/Claude can't:*
    - ✅ Real-time price comparison across retailers
    - ✅ Pattern-based fake review detection
    - ✅ Price history analysis for deal validation
    - ✅ Multi-agent collaboration with specialized roles
    - ✅ Personalized recommendations based on your priorities

    **Framework:** [CrewAI](https://github.com/crewAIInc/crewAI) - Role-based AI Agent Orchestration
    """)


if __name__ == "__main__":
    main()
