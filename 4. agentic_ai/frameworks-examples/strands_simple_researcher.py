"""
Simple Strands Research Agent - Modern AI Pro Workshop
Framework: Strands Agents SDK (AWS)
Simplified version for learning the basics

Key Concept: Multi-agent orchestration
- 2 agents working together
- Web search capabilities
- Simple research workflow
"""

import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Strands framework
from strands import Agent, tool
from duckduckgo_search import DDGS

load_dotenv()

# ============== CUSTOM TOOLS ==============

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information"""
    try:
        results = DDGS().text(query, region='wt-wt', max_results=max_results)
        search_results = []
        for r in results:
            search_results.append({
                "title": r.get("title", ""),
                "content": r.get("body", ""),
                "url": r.get("href", "")
            })
        return json.dumps(search_results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def get_news(topic: str, max_results: int = 3) -> str:
    """Get recent news on a topic"""
    try:
        results = DDGS().news(topic, region='wt-wt', max_results=max_results)
        news_items = []
        for r in results:
            news_items.append({
                "title": r.get("title", ""),
                "date": r.get("date", ""),
                "summary": r.get("body", "")[:300]
            })
        return json.dumps(news_items, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

# ============== AGENTS ==============

def create_researcher_agent():
    """Create a researcher agent that gathers information"""
    system_prompt = """You are a Research Agent.
Your job: Search for information on the given topic.
- Use web_search to find comprehensive info
- Use get_news to find recent news
- Summarize key findings with sources
- Be concise and factual"""

    return Agent(
        system_prompt=system_prompt,
        tools=[web_search, get_news]
    )

def create_summarizer_agent():
    """Create a summarizer agent that synthesizes findings"""
    system_prompt = """You are a Summarizer Agent.
Your job: Take research findings and create a clear summary.
- Extract key facts
- Organize by topic
- Highlight the most important information
- Use markdown formatting
- Be concise (3-4 paragraphs max)"""

    return Agent(
        system_prompt=system_prompt,
        tools=[]  # No tools needed for summarization
    )

# ============== WORKFLOW ==============

def run_research(topic: str):
    """
    Simple 2-phase research workflow
    Phase 1: Gather information
    Phase 2: Synthesize findings
    """
    results = {
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "findings": {},
        "summary": ""
    }

    # Phase 1: Research
    st.info("📚 Phase 1: Gathering information...")
    researcher = create_researcher_agent()

    research_prompt = f"""Research this topic thoroughly: {topic}

Please:
1. Search for key information
2. Find recent news or developments
3. List the most important facts"""

    research_output = researcher(research_prompt)
    results["findings"] = str(research_output)

    # Phase 2: Summarize
    st.info("📝 Phase 2: Summarizing findings...")
    summarizer = create_summarizer_agent()

    summary_prompt = f"""Create a brief summary from these research findings:

{results['findings']}

Write a clear, concise summary (3-4 paragraphs) suitable for quick reading."""

    summary_output = summarizer(summary_prompt)
    results["summary"] = str(summary_output)

    return results

# ============== STREAMLIT UI ==============

def main():
    st.set_page_config(
        page_title="Simple Strands Researcher",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 Simple Strands Research Agent")
    st.caption("Multi-agent research using Strands SDK")

    # Sidebar info
    with st.sidebar:
        st.header("How It Works")
        st.markdown("""
        ### Two-Agent System

        **1️⃣ Researcher Agent**
        - Searches the web
        - Finds recent news
        - Gathers raw data

        **2️⃣ Summarizer Agent**
        - Takes raw findings
        - Extracts key facts
        - Creates clear summary

        ### Why Strands?
        - Multi-agent orchestration
        - Specialized agent roles
        - Tool integration
        - Event-driven flow
        """)

    # Main content
    st.subheader("Enter a Research Topic")

    # Example topics
    examples = [
        "What are the latest AI developments?",
        "How is climate change being addressed?",
        "What's new in space exploration?",
        "Latest trends in renewable energy"
    ]

    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input(
            "Topic to research:",
            placeholder="Enter any topic..."
        )

    with col2:
        if st.button("Search", type="primary", use_container_width=True):
            if topic:
                st.session_state['current_topic'] = topic
                st.session_state['run_research'] = True

    # Show examples
    st.divider()
    st.markdown("**Try these examples:**")
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, use_container_width=True):
                st.session_state['current_topic'] = example
                st.session_state['run_research'] = True

    # Run research if triggered
    if st.session_state.get('run_research'):
        st.divider()

        with st.spinner("Running research workflow..."):
            results = run_research(st.session_state['current_topic'])

        st.session_state['results'] = results
        st.session_state['run_research'] = False

    # Display results
    if 'results' in st.session_state:
        results = st.session_state['results']

        st.success("✅ Research Complete!")

        # Tabs for results
        tab1, tab2, tab3 = st.tabs([
            "📋 Summary",
            "🔍 Raw Findings",
            "ℹ️ About"
        ])

        with tab1:
            st.markdown("## Summary")
            st.markdown(results['summary'])

        with tab2:
            st.markdown("## Research Findings")
            with st.expander("Show findings"):
                st.json(results['findings'])

        with tab3:
            st.markdown("""
            ### What This Demonstrates

            **Multi-Agent Pattern:**
            - Agent 1 (Researcher) specializes in gathering data
            - Agent 2 (Summarizer) specializes in synthesis
            - Agents work in sequence to accomplish complex task

            **Tools:**
            - Researcher uses web_search and get_news tools
            - Summarizer uses pure reasoning

            **Why This Matters:**
            Each agent has a focused role, making the system:
            - More reliable (specialized agents)
            - More transparent (see each phase)
            - More extensible (easy to add agents)

            This is more powerful than a single LLM because:
            - Separation of concerns
            - Better orchestration
            - Easier to debug and improve
            """)

    # Footer
    st.divider()
    st.caption(
        "🎓 Modern AI Pro Workshop | "
        "[Strands SDK](https://github.com/strands-agents/sdk-python)"
    )

if __name__ == "__main__":
    main()
