from crewai import Agent
import streamlit as st
import streamlit as st
import asyncio
import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
from langchain_community.llms import OpenAI
from langchain_community.tools import DuckDuckGoSearchRun
duckduckgo_search = DuckDuckGoSearchRun()
from src.tools.browser import BrowserTools
from src.tools.calculator import CalculatorTools
from src.tools.search import SearchTools
import os
import google.generativeai as genai
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import warnings
warnings.filterwarnings('ignore')
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from crewai import Agent, Task, Crew, Process
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


llm= ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.8, google_api_key=google_api_key)
def streamlit_callback(step_output):
    # This function will be called after each step of the agent's execution
    st.markdown("---")
    for step in step_output:
        if isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            if isinstance(action, dict) and "tool" in action and "tool_input" in action and "log" in action:
                st.markdown(f"# Action")
                st.markdown(f"**Tool:** {action['tool']}")
                st.markdown(f"**Tool Input** {action['tool_input']}")
                st.markdown(f"**Log:** {action['log']}")
                st.markdown(f"**Action:** {action['Action']}")
                st.markdown(
                    f"**Action Input:** ```json\n{action['tool_input']}\n```")
            elif isinstance(action, str):
                st.markdown(f"**Action:** {action}")
            else:
                st.markdown(f"**Action:** {str(action)}")

            st.markdown(f"**Observation**")
            if isinstance(observation, str):
                observation_lines = observation.split('\n')
                for line in observation_lines:
                    if line.startswith('Title: '):
                        st.markdown(f"**Title:** {line[7:]}")
                    elif line.startswith('Link: '):
                        st.markdown(f"**Link:** {line[6:]}")
                    elif line.startswith('Snippet: '):
                        st.markdown(f"**Snippet:** {line[9:]}")
                    elif line.startswith('-'):
                        st.markdown(line)
                    else:
                        st.markdown(line)
            else:
                st.markdown(str(observation))
        else:
            st.markdown(step)


class TripAgents():

    def city_selection_agent(self):
        return Agent(
            role='City Selection Expert',
            goal='Select the best city based on weather, season, and prices',
            backstory='An expert in analyzing travel data to pick ideal destinations',
            tools=[
                SearchTools.search_internet,
                duckduckgo_search,
            ],
            verbose=True,
            step_callback=streamlit_callback,llm=llm
        )

    def local_expert(self):
        return Agent(
            role='Local Expert at this city',
            goal='Provide the BEST insights about the selected city',
            backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
            tools=[
                SearchTools.search_internet,
                duckduckgo_search,
            ],
            verbose=True,
            step_callback=streamlit_callback,llm=llm
        )

    def travel_concierge(self):
        return Agent(
            role='Amazing Travel Concierge',
            goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
            backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
            tools=[
                SearchTools.search_internet,
                duckduckgo_search,
                CalculatorTools.calculate,
            ],
            verbose=True,
            step_callback=streamlit_callback,llm=llm
        )