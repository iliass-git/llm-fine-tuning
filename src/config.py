class Config:
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    DATASET_PATH = "/content/dataset/data-set.jsonl"
    TEXT_FIELD = "text"
    USE_DATASET_SUBSET = False
    SUBSET_SIZE = 1000
    OUTPUT_DIR = "./mistral-lora-output"
    MERGED_OUTPUT_DIR = "./mistral-merged-model"
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    MAX_SEQ_LENGTH = 512
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
