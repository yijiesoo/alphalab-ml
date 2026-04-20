#!/usr/bin/env python
"""Entry point: run the full factor-ML pipeline end-to-end."""

import logging
import sys
from pathlib import Path

# Allow running directly from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphalab_ml.pipeline import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run(config_path)
