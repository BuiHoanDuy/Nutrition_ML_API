#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import uvicorn

# Set environment variables for proper encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import the app
from api.main import app

if __name__ == "__main__":
    # Run the server with proper configuration
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
