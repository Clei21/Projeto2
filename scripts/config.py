import os
import random

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SPIDER_DIR = os.path.join(DATA_DIR, "spider")
SPIDER_DB_DIR = os.path.join(SPIDER_DIR, "database")
SPIDER_TRAIN_FILE = os.path.join(SPIDER_DIR, "train_spider.json")
SPIDER_DEV_FILE = os.path.join(SPIDER_DIR, "dev.json")
SPIDER_TABLES_FILE = os.path.join(SPIDER_DIR, "tables.json")

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
SEED = int(os.environ.get("SEED", "42"))

MMLU_SUBJECTS = {
    "STEM": "college_computer_science",
    "Humanities": "philosophy",
    "Social Sciences": "high_school_macroeconomics",
}
MMLU_QUESTIONS_PER_CATEGORY = 50
MMLU_NUM_SHOTS = 5

MAX_NEW_TOKENS_SQL = 256
MAX_NEW_TOKENS_MMLU = 8


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
