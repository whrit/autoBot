#!/bin/bash
cd /home/bw/Projects/autoBot
PYTHONPATH=services/ingestor_py .venv/bin/pytest services/ingestor_py/tests/ -v 2>&1
