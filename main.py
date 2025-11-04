from utils import check_gpu, install_dependencies, clear_gpu_memory
from config import Config
from model import get_model_configs, load_tokenizer, load_model, train_model, merge_and_save_model
from data import load_dataset
from test import test_model


def main():
    check_gpu()
    install_dependencies()
    clear_gpu_memory()
    lora_config, bnb_config, training_args = get_model_configs()
    tokenizer = load_tokenizer(Config.MODEL_NAME)
    model = load_model(Config.MODEL_NAME, bnb_config, lora_config)
    dataset = load_dataset(
        Config.DATASET_PATH,
        Config.TEXT_FIELD,
        Config.USE_DATASET_SUBSET,
        Config.SUBSET_SIZE
    )
    trainer = train_model(model, dataset, tokenizer, training_args, Config.TEXT_FIELD)
    merged_model = merge_and_save_model(
        trainer,
        Config.MODEL_NAME,
        Config.OUTPUT_DIR,
        Config.MERGED_OUTPUT_DIR
    )
    test_prompts = ["Type your test prompt here"]
    test_model(merged_model, tokenizer, test_prompts)
    print("\n✅ Training complete!")
    print(f"Model saved to: {Config.MERGED_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
