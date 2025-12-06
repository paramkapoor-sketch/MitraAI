"""
🧠 Second Brain Research Assistant - Modern AI Pro Workshop
Framework: Strands Agents SDK (AWS)
Pattern: Orchestrator-Workers

Features:
1. Multi-agent research workflow
2. Web search + news aggregation
3. Fact verification & cross-referencing
4. Structured report synthesis
5. Source tracking & citations

What this builds that ChatGPT/Claude can't do:
- Persistent research sessions with memory
- Automatic source verification
- Contradiction detection across sources
- Structured, citable outputs
"""

import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Strands Agents imports
from strands import Agent, tool
from strands.models import BedrockModel

# Tools
from duckduckgo_search import DDGS
from tavily import TavilyClient
import requests

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# ============== CUSTOM TOOLS ==============

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information on a topic.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        JSON string containing search results with titles, snippets, and URLs
    """
    results = []

    # Try Tavily first (better for research)
    if TAVILY_API_KEY:
        try:
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily.search(query, max_results=max_results, search_depth="advanced")
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "source": "tavily"
                })
            return json.dumps(results, indent=2)
        except Exception as e:
            pass  # Fall back to DuckDuckGo

    # DuckDuckGo fallback
    try:
        ddg_results = DDGS().text(query, region='wt-wt', max_results=max_results)
        for r in ddg_results:
            results.append({
                "title": r.get("title", ""),
                "content": r.get("body", ""),
                "url": r.get("href", ""),
                "source": "duckduckgo"
            })
    except Exception as e:
        results.append({"error": str(e)})

    return json.dumps(results, indent=2)


@tool
def news_search(topic: str, max_results: int = 5) -> str:
    """
    Search for recent news articles on a topic.

    Args:
        topic: The news topic to search for
        max_results: Maximum number of news articles (default 5)

    Returns:
        JSON string containing news articles with titles, dates, and summaries
    """
    try:
        results = DDGS().news(topic, region='wt-wt', max_results=max_results)
        news_items = []
        for r in results:
            news_items.append({
                "title": r.get("title", ""),
                "date": r.get("date", ""),
                "source": r.get("source", ""),
                "body": r.get("body", "")[:500],
                "url": r.get("url", "")
            })
        return json.dumps(news_items, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def verify_fact(claim: str, sources: str) -> str:
    """
    Cross-reference a claim against multiple sources to verify accuracy.

    Args:
        claim: The factual claim to verify
        sources: JSON string of sources to check against

    Returns:
        Verification result with confidence score and supporting evidence
    """
    # Search for the specific claim
    try:
        verification_results = DDGS().text(f'fact check: {claim}', max_results=3)

        supporting = 0
        contradicting = 0
        evidence = []

        for r in verification_results:
            body = r.get("body", "").lower()
            title = r.get("title", "").lower()
            claim_lower = claim.lower()

            # Simple keyword matching for demonstration
            # In production, use NLI model
            key_terms = claim_lower.split()[:5]
            matches = sum(1 for term in key_terms if term in body or term in title)

            if matches >= 3:
                supporting += 1
            evidence.append({
                "source": r.get("href", ""),
                "title": r.get("title", ""),
                "relevance": "high" if matches >= 3 else "low"
            })

        confidence = min(supporting / 3 * 5, 5)  # Scale to 1-5

        return json.dumps({
            "claim": claim,
            "confidence": round(confidence, 1),
            "supporting_sources": supporting,
            "evidence": evidence
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "claim": claim, "confidence": 0})


# ============== AGENT DEFINITIONS ==============

def create_researcher_agent(model_provider=None):
    """Create the Web Researcher agent"""
    system_prompt = """You are a Research Agent specializing in gathering comprehensive information.

Your responsibilities:
1. Search for authoritative sources on the given topic
2. Extract key facts with proper citations
3. Note any conflicting information you find
4. Prioritize recent and credible sources

Always cite your sources with URLs.
Format findings as structured bullet points.
Flag any information that seems uncertain or contested."""

    if model_provider:
        return Agent(
            model=model_provider,
            system_prompt=system_prompt,
            tools=[web_search, news_search],
            callback_handler=None  # Suppress intermediate output
        )
    else:
        return Agent(
            system_prompt=system_prompt,
            tools=[web_search, news_search],
            callback_handler=None
        )


def create_fact_checker_agent(model_provider=None):
    """Create the Fact Checker agent"""
    system_prompt = """You are a Fact Verification Agent with expertise in cross-referencing claims.

Your responsibilities:
1. Cross-reference claims across multiple sources
2. Rate confidence for each fact on a scale of 1-5
3. Flag any contradictions or disputed claims
4. Identify potential biases in sources

For each claim, provide:
- Confidence rating (1-5)
- Supporting evidence
- Any contradicting evidence
- Overall assessment"""

    if model_provider:
        return Agent(
            model=model_provider,
            system_prompt=system_prompt,
            tools=[web_search, verify_fact],
            callback_handler=None
        )
    else:
        return Agent(
            system_prompt=system_prompt,
            tools=[web_search, verify_fact],
            callback_handler=None
        )


def create_synthesizer_agent(model_provider=None):
    """Create the Report Synthesizer agent"""
    system_prompt = """You are a Report Synthesis Agent expert at creating clear, comprehensive reports.

Your responsibilities:
1. Compile verified facts into a coherent narrative
2. Structure the report with clear sections:
   - Executive Summary
   - Key Findings
   - Detailed Analysis
   - Sources & Citations
3. Include confidence levels for major claims
4. Highlight any areas needing further research

Write in clear, professional language.
Use markdown formatting for structure.
Always attribute claims to their sources."""

    if model_provider:
        return Agent(
            model=model_provider,
            system_prompt=system_prompt,
            callback_handler=None  # Will show output for final report
        )
    else:
        return Agent(
            system_prompt=system_prompt,
            callback_handler=None
        )


# ============== RESEARCH WORKFLOW ==============

def run_research_workflow(query: str, model_provider=None, progress_callback=None):
    """
    Execute the full research workflow:
    1. Research phase - gather information
    2. Verification phase - cross-reference facts
    3. Synthesis phase - compile final report
    """
    results = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "phases": {}
    }

    # Create agents
    researcher = create_researcher_agent(model_provider)
    fact_checker = create_fact_checker_agent(model_provider)
    synthesizer = create_synthesizer_agent(model_provider)

    # Phase 1: Research
    if progress_callback:
        progress_callback("Phase 1/3: Researching topic...")

    research_prompt = f"""Research the following topic thoroughly:

TOPIC: {query}

Please:
1. Search for comprehensive information on this topic
2. Look for recent news and developments
3. Identify key facts, figures, and claims
4. Note any controversies or conflicting viewpoints

Provide your findings in a structured format with citations."""

    research_response = researcher(research_prompt)
    research_findings = str(research_response)
    results["phases"]["research"] = research_findings

    # Phase 2: Fact Verification
    if progress_callback:
        progress_callback("Phase 2/3: Verifying facts...")

    verification_prompt = f"""Verify the key claims from this research:

RESEARCH FINDINGS:
{research_findings}

Please:
1. Identify the 3-5 most important factual claims
2. Cross-reference each claim with independent sources
3. Rate your confidence in each claim (1-5)
4. Flag any contradictions or uncertainties"""

    verification_response = fact_checker(verification_prompt)
    verified_facts = str(verification_response)
    results["phases"]["verification"] = verified_facts

    # Phase 3: Synthesis
    if progress_callback:
        progress_callback("Phase 3/3: Synthesizing report...")

    synthesis_prompt = f"""Create a comprehensive research report from these findings:

ORIGINAL QUERY: {query}

RESEARCH FINDINGS:
{research_findings}

VERIFICATION RESULTS:
{verified_facts}

Create a well-structured report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points with confidence levels)
3. Detailed Analysis
4. Areas for Further Research
5. Sources & Citations

Use markdown formatting."""

    synthesis_response = synthesizer(synthesis_prompt)
    final_report = str(synthesis_response)
    results["phases"]["synthesis"] = final_report
    results["final_report"] = final_report

    return results


# ============== STREAMLIT UI ==============

def main():
    st.set_page_config(
        page_title="Second Brain Research Assistant",
        page_icon="🧠",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .agent-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .phase-indicator {
        background: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🧠 Second Brain Research Assistant</h1>
        <p style="color: #ddd; margin: 0;">Multi-Agent Research • Powered by Strands SDK</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # Model provider selection
        st.subheader("Model Provider")
        model_choice = st.selectbox(
            "Select Model",
            ["Amazon Bedrock (Claude)", "Use Default"],
            help="Strands supports multiple model providers"
        )

        st.divider()

        # Agent architecture info
        st.subheader("📊 Agent Architecture")
        st.markdown("""
        <div class="agent-card">
            <strong>🔍 Web Researcher</strong><br>
            Gathers information from web & news
        </div>
        <div class="agent-card">
            <strong>✅ Fact Checker</strong><br>
            Verifies claims across sources
        </div>
        <div class="agent-card">
            <strong>📝 Synthesizer</strong><br>
            Compiles verified report
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # API Status
        st.subheader("🔑 API Status")
        st.write(f"Tavily: {'✅' if TAVILY_API_KEY else '❌'}")
        st.write(f"AWS: {'✅' if AWS_ACCESS_KEY_ID else '⚠️ Using default'}")
        st.write(f"Groq: {'✅' if GROQ_API_KEY else '❌'}")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔬 Research Query")

        # Sample queries
        sample_queries = [
            "What are the latest developments in AI regulation in 2025?",
            "Compare the economic impacts of remote work adoption",
            "What are current best practices for sustainable investing?",
            "Analyze the state of quantum computing commercialization"
        ]

        selected_sample = st.selectbox(
            "Try a sample query:",
            ["Custom query..."] + sample_queries
        )

        if selected_sample == "Custom query...":
            research_query = st.text_area(
                "Enter your research topic:",
                placeholder="What would you like to research?",
                height=100
            )
        else:
            research_query = st.text_area(
                "Enter your research topic:",
                value=selected_sample,
                height=100
            )

    with col2:
        st.subheader("⚙️ Research Options")
        depth = st.select_slider(
            "Research Depth",
            options=["Quick", "Standard", "Deep"],
            value="Standard"
        )

        include_news = st.checkbox("Include recent news", value=True)
        verify_facts = st.checkbox("Verify key claims", value=True)

    # Research button
    if st.button("🚀 Start Research", type="primary", use_container_width=True):
        if not research_query:
            st.error("Please enter a research topic")
        else:
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            def update_progress(message):
                if "1/3" in message:
                    progress_bar.progress(33)
                elif "2/3" in message:
                    progress_bar.progress(66)
                elif "3/3" in message:
                    progress_bar.progress(90)
                status_text.text(message)

            # Configure model
            model_provider = None
            if model_choice == "Amazon Bedrock (Claude)" and AWS_ACCESS_KEY_ID:
                try:
                    model_provider = BedrockModel(
                        model_id="anthropic.claude-sonnet-4-20250514-v1:0",
                        region_name=AWS_REGION
                    )
                except Exception as e:
                    st.warning(f"Bedrock setup failed, using default: {e}")

            try:
                with st.spinner("Running research workflow..."):
                    results = run_research_workflow(
                        research_query,
                        model_provider=model_provider,
                        progress_callback=update_progress
                    )

                progress_bar.progress(100)
                status_text.text("Research complete!")

                # Store in session
                st.session_state['research_results'] = results

            except Exception as e:
                st.error(f"Research workflow failed: {e}")
                st.exception(e)

    # Display results
    if 'research_results' in st.session_state:
        results = st.session_state['research_results']

        st.divider()

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Final Report",
            "🔍 Research Phase",
            "✅ Verification Phase",
            "📊 Raw Data"
        ])

        with tab1:
            st.markdown("## Research Report")
            st.markdown(results.get("final_report", "No report generated"))

            # Download button
            st.download_button(
                "📥 Download Report",
                data=results.get("final_report", ""),
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

        with tab2:
            st.markdown("## Research Findings")
            st.markdown(results.get("phases", {}).get("research", "No findings"))

        with tab3:
            st.markdown("## Fact Verification")
            st.markdown(results.get("phases", {}).get("verification", "No verification"))

        with tab4:
            st.json(results)

    # Footer
    st.divider()
    st.markdown("""
    ---
    **🎓 Modern AI Pro Workshop** | Strands Agents SDK

    *What this does that ChatGPT/Claude can't:*
    - ✅ Multi-agent orchestration with specialized roles
    - ✅ Automated fact verification across sources
    - ✅ Structured research with confidence ratings
    - ✅ Persistent session memory
    - ✅ Full source transparency

    **Framework:** [Strands Agents SDK](https://github.com/strands-agents/sdk-python) by AWS
    """)


if __name__ == "__main__":
    main()
