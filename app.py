import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools import DuckDuckGoSearchRun

# --- CONFIGURATION ---
MY_KEY = "AIzaSyB8G8KFruLkGrPUXOGRSSeKXfXVsanEobs"
gemini_llm = LLM(model="gemini/gemini-1.5-flash", api_key=MY_KEY)

# --- INTERFACE HEADER ---
st.set_page_config(page_title="AI Travel Agency", layout="wide")
st.title("🌍 Elite AI Travel Concierge")
st.markdown("Enter your trip details below to have the AI agents build your custom itinerary.")

# --- SIDEBAR INPUTS (The New Interface) ---
with st.sidebar:
    st.header("📍 Trip Logistics")
    location = st.text_input("Destination", placeholder="e.g. Newport, RI")
    start_loc = st.text_input("Starting From", placeholder="e.g. New York, NY")
    budget = st.number_input("Total Budget ($)", min_value=500, value=2000, step=100)
    days = st.slider("Trip Length (Days)", 1, 14, 3)
    
    is_road_trip = st.toggle("Is this a Road Trip?")
    mode = st.selectbox("Mode of Travel", ["Own Car", "Rental", "Flight", "Train"])
    
    st.header("🎗️ Preferences")
    memberships = st.text_input("Memberships", placeholder="e.g. AAA, Marriott")
    extra_details = st.text_area("Wants & Needs (Paragraph)", placeholder="I love quiet rooms and dark-academia coffee...")

# --- GENERATION LOGIC ---
if st.button("🚀 Generate Comprehensive Travel Plan"):
    if not location:
        st.error("Please enter a destination!")
    else:
        with st.spinner(f"Agents are researching {location}... This usually takes 60-90 seconds."):
            
            # Initialize Tools & Agents
            search_tool = DuckDuckGoSearchRun()
            scrape_tool = ScrapeWebsiteTool()
            
            scout = Agent(role='Travel Scout', goal='Find hotel choices and prices.', tools=[search_tool, scrape_tool], llm=gemini_llm)
            
            # The Task
            main_task = Task(
                description=(
                    f"Plan a {days} day trip to {location} from {start_loc}. Budget: ${budget}. "
                    f"Road Trip: {is_road_trip}. Mode: {mode}. Memberships: {memberships}. "
                    f"Specific Needs: {extra_details}. Calculate all-in prices (tax/tips)."
                ),
                expected_output="A detailed report with 3 hotels, links, and a full daily itinerary.",
                agent=scout
            )

            # Run the Crew
            crew = Crew(agents=[scout], tasks=[main_task])
            result = crew.kickoff()

            # --- DISPLAY RESULTS ---
            st.success("✅ Research Complete!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("🗺️ Your Custom Itinerary")
                st.write(str(result))
                
            with col2:
                st.header("💰 Summary Statistics")
                st.metric("Destination", location)
                st.metric("Budget Remaining", f"${budget}")
                
            # Download Button
            st.download_button(
                label="📥 Download Plan as Text",
                data=str(result),
                file_name=f"Trip_Plan_{location}.txt",
                mime="text/plain"
            )