"""Create a logger for the edflib package."""

import os
import json
import logging.config
import importlib.resources

if not os.path.exists("logs"):
    os.mkdir("logs")

with importlib.resources.read_text("edflib", "logging_config.json") as f:
    config = json.load(f)

logging.config.dictConfig(config)
logger = logging.getLogger(__name__)
