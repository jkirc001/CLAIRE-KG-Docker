"""
CLAIRE Knowledge Graph - Cybersecurity Data Ingestion and Analysis
"""

__version__ = "0.1.0"

# Export commonly used utilities
from .question_classifier import is_heavy_question, is_out_of_domain

__all__ = ["is_heavy_question", "is_out_of_domain"]
