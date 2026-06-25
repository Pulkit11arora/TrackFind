Skip to content
Pulkit11arora
TrackFind
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security and quality
Insights
Settings
Files
Go to file
t
T
.devcontainer
app.py
requirements.txt
TrackFind
/
app.py
in
main

Edit

Preview
Indent mode

Spaces
Indent size

4
Line wrap mode

No wrap
Editing app.py file contents
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
"""
TrackFind — Your Personal AI Music Curator
============================================
A premium, dark-themed Streamlit music recommendation app powered by the
Google Gen AI SDK (Gemini).

import os
import re
import csv
import io
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Google Gen AI SDK
# ---------------------------------------------------------------------------
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ===========================================================================
# 1. CONFIGURATION & CONSTANTS
# ===========================================================================

APP_TITLE = "🎵 TrackFind — Your Personal AI Music Curator"

# Google retires Gemini model IDs on a rolling basis — gemini-1.5-flash and
# gemini-2.0-flash have both already been shut down (404 NOT_FOUND on any
# request). gemini-2.5-flash is the current widely-available default as of
# June 2026. If you hit a 404 again in the future, just change this one
# string — everything else in the app is model-agnostic.
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
