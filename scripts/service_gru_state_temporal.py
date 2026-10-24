"""Run independent Normal-state and original-GRU max fusion."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.service_gru_error_calibration import main

if __name__ == "__main__":
    main(required_schema="service-gru-state-temporal-v1")
