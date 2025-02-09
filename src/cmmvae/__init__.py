"""
.. include:: ../../INSTALLATION.md
.. include:: ../../USAGE.md
.. include:: ../../CONTRIBUTING.md
"""

from . import (
    callbacks,
    cli,
    config,
    constants,
    data,
    models,
    modules,
    utils,
)

import logging
import os
import yaml
from logging.config import dictConfig

def setup_logging():
    LOG_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "logging_config.yaml")
    if os.path.exists(LOG_CONFIG_FILE):
        with open(LOG_CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)
            dictConfig(config)

setup_logging()
logger = logging.getLogger("cmmvae")
logger.setLevel(os.getenv("CMMVAE_LOG_LEVEL", "INFO").upper())
logger.debug("CMMVAE package initialized")