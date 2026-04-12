"""Vercel Python runtime entrypoint for RiskLab Monte Carlo Simulator."""
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app_dir = os.path.join(root_dir, 'app')
sys.path.insert(0, root_dir)
sys.path.insert(0, app_dir)

os.environ['STREAMLIT_SERVER_PORT'] = os.environ.get('PORT', '8000')
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

import main

from http.server import BaseHTTPRequestHandler

def handler(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b'RiskLab Monte Carlo Portfolio Simulator is running.']

app = handler
