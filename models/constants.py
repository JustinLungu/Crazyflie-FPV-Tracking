########################################## Model/Dataset Location Constants ###############################

# Root for class folders containing YOLO-ready datasets:
# data/labels/<class_name>/<dataset_name>/
# Keep class/dataset names aligned with data/prepare_yolo_dataset.py output.
YOLO_LABELS_ROOT = "data/labels"
YOLO_TARGET_CLASS_NAME = "black_drone"
YOLO_OUTPUT_DATASET_NAME = "black_drone_yolo"
YOLO_DATASET_YAML_NAME = "dataset.yaml"


########################################## Training Constants #############################################

# Base model to fine-tune.
YOLO_TRAIN_MODEL = "yolo26s.pt"

# Hardware/performance settings.
YOLO_IMG_SIZE = 960
YOLO_EPOCHS = 150
YOLO_BATCH = 4
YOLO_DEVICE = 0
YOLO_WORKERS = 2
YOLO_PATIENCE = 30
YOLO_CACHE_IMAGES = False

# Output folder structure under runs/:
# runs/models/<train_label>_<date>/
# runs/comparison/<comparison_label>_<date>/...
YOLO_RUNS_ROOT = "runs"
YOLO_MODELS_RUNS_DIR = "models"
YOLO_COMPARISON_RUNS_DIR = "comparison"
YOLO_RUN_DATE_FORMAT = "%Y%m%d_%H%M%S"

# Base labels used for dated folder names.
YOLO_TRAIN_RUN_LABEL = "black_drone_yolo26s"


########################################## Test/Eval Constants ############################################

# Which trained weights to evaluate on the test split.
# Accepted values:
# - "latest_best": latest run in runs/models/*, best.pt
# - "latest_last": latest run in runs/models/*, last.pt
# - explicit local path to .pt
# - Ultralytics model alias (e.g. yolo26s.pt)
YOLO_TEST_WEIGHTS = "latest_best"
YOLO_TEST_SPLIT = "test"  # "val" or "test"
YOLO_TEST_BATCH = 4
YOLO_TEST_CONF = 0.001
YOLO_TEST_IOU = 0.7
YOLO_TEST_RUN_LABEL = "black_drone_eval"


########################################## Multi-Model Comparison Constants ###############################

# Evaluate all listed models on the same dataset split and print one summary table.
YOLO_COMPARE_MODEL_REFS = (
    YOLO_TEST_WEIGHTS,
    "latest_last",
    "yolo26n.pt",
    "yolo26s.pt",
)
YOLO_COMPARE_RUN_LABEL = "black_drone_compare"
