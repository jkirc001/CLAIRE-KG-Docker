#!/usr/bin/env python3
"""
CLAIRE-KG CLI entry point. Run with: python -m claire_kg (delegates to cli.app).
"""

from .cli import app

if __name__ == "__main__":
    app()
