# Mistral LoRA Fine-Tuning (Local Dataset)

This project provides a script for fine-tuning the Mistral-7B model using LoRA (Low-Rank Adaptation) on a local JSONL dataset, optimized for GPU systems.

## Features
- Fine-tunes Mistral-7B with LoRA adapters
- Supports quantization for efficient training
- Uses HuggingFace Transformers, PEFT, and TRL libraries
- Loads and processes local JSONL datasets
- Includes model merging and testing utilities

## Requirements
- GPU-enabled system (NVIDIA CUDA)
- Python 3.8+
- It is recommended to use a Python virtual environment:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate
  ```
- Install dependencies:
  ```powershell
  pip install -r requirements.txt
  ```

## Usage
1. Place your dataset in JSONL format at the path specified in `Config.DATASET_PATH` (default: `/content/dataset/data-set.jsonl`).
2. (Recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the script:
   ```powershell
   python main.py
   ```
5. The script will:
   - Check GPU availability
   - Load the model and dataset
   - Train with LoRA
   - Merge and save the final model
   - Test the model with a sample prompt

## Output
- LoRA adapter model: `./mistral-lora-output`
- Merged final model: `./mistral-merged-model`

## Customization
Edit the `Config` class in `main.py` to change model, dataset path, training parameters, and output directories.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file or visit https://www.apache.org/licenses/LICENSE-2.0 for details.
