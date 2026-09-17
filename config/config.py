"""Central configuration for the project.

Keeping every tunable knob in one place makes the experiment
reproducible and easy to change without hunting through scripts.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "metrics.json"

# --- Data ----------------------------------------------------------------
TICKER = "AAPL"          # primary ticker for the main experiment
START_DATE = "2015-01-01"
END_DATE = None          # None = up to today
INTERVAL = "1d"          # daily bars

# --- Features --------------------------------------------------------------
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COLUMN = "Close"

# --- Sliding window --------------------------------------------------------
LOOKBACK = 60             # trading days of history fed into the model
HORIZON = 1               # predict this many days ahead (next-day close)

# --- Chronological split (must sum to 1.0) --------------------------------
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# --- Model -----------------------------------------------------------------
RANDOM_SEED = 42
