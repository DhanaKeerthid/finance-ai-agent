import streamlit as st
from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Combined Agent
finance_ai_agent = Agent(
    name="Financial AI Agent",
    role="Search the web and analyze financial data.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"), 
    tools=[
        DuckDuckGo(), 
        YFinanceTools(stock_price=True, analyst_recommendations=True, 
                     stock_fundamentals=True, company_news=True)
    ],
    instructions=[
        "If the question requires web search, use DuckDuckGo first.", 
        "If the question involves financial data, use YFinanceTools.", 
        "Always include sources when using external tools.", 
        "Use tables to display financial data whenever possible."
    ],
    show_tools_calls=True,
    markdown=True,
)

# Streamlit App UI
st.title("Financial AI Agent")

st.write("""
    This is a Financial AI Agent that can search the web and analyze financial data.
    You can ask it questions about stock prices, analyst recommendations, and more!
""")

# Input box for user question
user_query = st.text_input("Ask a financial question:")

if user_query:
    # Show loading spinner while the agent processes the query
    with st.spinner("Processing your request..."):
        # Running the agent and capturing the response
        response = finance_ai_agent.run(user_query)  # Use run() instead of print_response()
    
    # Check if the response is not None and display
    if response:
        st.write(response.content)
    else:
        st.write("No response generated. Please try again.")
