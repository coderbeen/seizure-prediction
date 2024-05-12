"""Create a logger for the edflib package."""

import os
import json
import logging.config
import importlib.resources

if not os.path.exists("logs"):
    os.mkdir("logs")

with importlib.resources.path("edflib", "logging_config.json") as path:
    with open(path, "r") as file:
        config = json.load(file)

logging.config.dictConfig(config)
logger = logging.getLogger(__name__)
