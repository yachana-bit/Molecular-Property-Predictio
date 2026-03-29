from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA = HERE / "data"

GCN_MODEL_PATH = HERE / "GCN_model.pt"
GIN_MODEL_PATH = HERE / "GIN_model.pt"

# ── Dataset ────────────────────────────────────────────────────────────────────
DATA_SIZE      = 30_000
TRAIN_RATIO    = 0.8
TEST_RATIO     = 0.1
VAL_RATIO      = 0.1
TARGET_COL     = 0          # index of QM9 regression target (0 = dipole moment μ)

# ── DataLoader ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 64

# ── Model ──────────────────────────────────────────────────────────────────────
GCN_DIM_H     = 128
GIN_DIM_H     = 64
DROPOUT_P     = 0.5
NUM_NODE_FEATURES = 11      # QM9 has 11 atom features

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS        = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 5e-4

# ── Visualization ──────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Atom type",
    "Charge",
    "Hybridization",
    "Aromatic",
    "H count",
    "In ring",
    "Chirality",
    "Mass",
    "Valence",
    "Radical electrons",
    "Implicit H",
]