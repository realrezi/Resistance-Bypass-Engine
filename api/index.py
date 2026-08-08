"""Vercel entrypoint for the FastAPI application."""

import os
import sys

# Add project root directory to sys.path for Vercel serverless functions.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.main import app

__all__ = ["app"]
