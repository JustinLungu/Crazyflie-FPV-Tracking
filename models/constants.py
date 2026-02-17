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

# Output run naming.
# Use "runs" (not "runs/detect") to avoid nested "runs/detect/runs/detect/...".
YOLO_PROJECT_DIR = "runs"
YOLO_TRAIN_RUN_NAME = "black_drone_yolo26s"


########################################## Test/Eval Constants ############################################

# Which trained weights to evaluate on the test split.
# Can be a local path (best.pt/last.pt) or a model alias.
YOLO_TEST_WEIGHTS = "runs/detect/runs/detect/black_drone_yolo26s/weights/best.pt"
YOLO_TEST_SPLIT = "test"  # "val" or "test"
YOLO_TEST_BATCH = 4
YOLO_TEST_CONF = 0.001
YOLO_TEST_IOU = 0.7
YOLO_TEST_RUN_NAME = "black_drone_yolo26s_test"


########################################## Multi-Model Comparison Constants ###############################

# Evaluate all listed models on the same dataset split and print one summary table.
YOLO_COMPARE_MODEL_REFS = (
    YOLO_TEST_WEIGHTS,
    "runs/detect/runs/detect/black_drone_yolo26s/weights/last.pt",
    "yolo26n.pt",
    "yolo26s.pt",
)
YOLO_COMPARE_RUN_PREFIX = "black_drone_compare"
