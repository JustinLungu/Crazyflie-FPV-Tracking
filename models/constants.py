########################################## Model/Dataset Location Constants ###############################

# Root for class folders containing YOLO-ready datasets:
# data/labels/<class_name>/<dataset_name>/
# Keep class/dataset names aligned with data/prepare_yolo_dataset.py output.
YOLO_LABELS_ROOT = "data/labels"
YOLO_TARGET_CLASS_NAME = "brushless_drone"
YOLO_OUTPUT_DATASET_NAME = YOLO_TARGET_CLASS_NAME + "_yolo"
YOLO_DATASET_YAML_NAME = "dataset.yaml"


########################################## Training Constants #############################################

# Base model to fine-tune.
YOLO_TRAIN_MODEL = "yolo26s.pt"

# Resume support:
# - False: start a fresh run with a new timestamped folder.
# - True: continue from YOLO_RESUME_WEIGHTS (usually latest_last).
YOLO_RESUME_TRAINING = False
# Accepted values:
# - "latest_last": resume most recent runs/models/*/weights/last.pt
# - explicit .pt path (usually .../weights/last.pt)
YOLO_RESUME_WEIGHTS = "runs/models/brushless_drone_yolo26s_20260331_172052/weights/last.pt"


# Hardware/performance settings.
# lower = faster, less VRAM, less detail; higher = better small-object detail, slower, more VRAM.
YOLO_IMG_SIZE = 1024 #960
YOLO_EPOCHS = 100
# lower = safer for VRAM, noisier gradients; higher = faster/steadier training, may OOM.
YOLO_BATCH = 8 
# compute target. 0 = first GPU; cpu = much slower; wrong GPU index fails.
YOLO_DEVICE = 0
# dataloader worker processes. Lower = less CPU/RAM load but possible data bottleneck; 
# higher = faster loading until CPU/IO saturates (too high can hurt stability).
YOLO_WORKERS = 2
# early-stop patience. Lower = stops sooner; higher = waits longer for improvements.
YOLO_PATIENCE = 10
# True can speed epochs after warmup, but uses much more RAM/disk cache.
YOLO_CACHE_IMAGES = False

# Output folder structure under runs/:
# runs/models/<train_label>_<date>/
# runs/evaluation/<eval_label>_<date>/
# runs/comparison/<comparison_label>_<date>/...
YOLO_RUNS_ROOT = "runs"
YOLO_MODELS_RUNS_DIR = "models"
YOLO_EVALUATION_RUNS_DIR = "evaluation"
YOLO_COMPARISON_RUNS_DIR = "comparison"
YOLO_RUN_DATE_FORMAT = "%Y%m%d_%H%M%S"

# Base labels used for dated folder names.
YOLO_TRAIN_RUN_LABEL = YOLO_TARGET_CLASS_NAME + "_yolo26s"


########################################## Test/Eval Constants ############################################

# Which trained weights to evaluate on the test split.
# Accepted values:
# - "latest_best": latest run in runs/models/*, best.pt
# - "latest_last": latest run in runs/models/*, last.pt
# - explicit local path to .pt (ex: runs/models/green_drone_yolo26s_20260215_170000/weights/best.pt)
# - Ultralytics model alias (e.g. yolo26s.pt)
YOLO_TEST_WEIGHTS = "runs/models/" + YOLO_TARGET_CLASS_NAME + "_yolo26s_20260410_045839/weights/best.pt"
YOLO_TEST_SPLIT = "test"  # "val" or "test"
YOLO_TEST_BATCH = 8
YOLO_TEST_CONF = 0.001
YOLO_TEST_IOU = 0.7
# Extra duplicate suppression after Ultralytics NMS for evaluation scripts.
# Rule: if overlap (intersection / smaller-box-area) exceeds this percent,
# keep only the higher-confidence box.
YOLO_EVAL_OVERLAP_SUPPRESSION_PERCENT = 30.0
YOLO_TEST_RUN_LABEL = YOLO_TARGET_CLASS_NAME + "_eval"

########################################## Prediction Preview Constants ###################################

# Random prediction grid from a dataset split (used by models/random_test_preview.py).
YOLO_PREVIEW_SPLIT = YOLO_TEST_SPLIT
YOLO_PREVIEW_IMAGE_COUNT = 9
# Number of preview grids to generate per run.
YOLO_PREVIEW_RUNS = 5
# Set int for reproducible sampling, or None for non-deterministic random picks.
YOLO_PREVIEW_RANDOM_SEED = None
# Optional output path relative to repo root:
# - directory path: saves one/many JPGs in that folder
# - file path (.jpg/.jpeg/.png/.bmp/.webp): used as basename (adds _01, _02, ... when runs > 1)
# - "" => auto-run folder in runs/evaluation
YOLO_PREVIEW_OUTPUT_PATH = "runs/evaluation/rand_preview/"


########################################## Multi-Model Comparison Constants ###############################

# Evaluate all listed models on the same dataset split and print one summary table.
YOLO_COMPARE_MODEL_REFS = (
    "runs/models/" + YOLO_TARGET_CLASS_NAME + "_yolo26s_20260322_221846/weights/best.pt",
    "runs/models/" + YOLO_TARGET_CLASS_NAME + "_yolo26s_20260331_172052/weights/best.pt",
    "runs/models/backup/weights/best.pt",

)
YOLO_COMPARE_RUN_LABEL = YOLO_TARGET_CLASS_NAME + "_compare"
