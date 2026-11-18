#!/usr/bin/env python3
"""Re-derives the computed numbers of the paper's tables (exact, no simulation)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from greencav.paper_checks import main
raise SystemExit(0 if main() else 1)
