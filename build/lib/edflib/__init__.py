"""Create a logger for the edflib package."""

import os
import json
import logging.config

if not os.path.exists("logs"):
    os.mkdir("logs")

with open("edflib/logging_config.json", "r") as f:
    config = json.load(f)
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)
