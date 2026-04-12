"""RiskLab - Monte Carlo Portfolio Simulator (Vercel entry point)."""
import os
import sys

# Add the app directory to the path so imports work correctly
app_dir = os.path.join(os.path.dirname(__file__), 'app')
sys.path.insert(0, app_dir)

# Import and run the main Streamlit app
from main import *
