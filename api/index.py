import os
import sys

# Add project root directory to sys.path for Vercel serverless functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app  # noqa: F401 - Vercel discovers this ASGI export.
