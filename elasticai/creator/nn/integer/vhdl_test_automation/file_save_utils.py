import logging
from pathlib import Path


def save_quant_data(tensor, file_dir: Path, file_name: str):
    if file_dir is None:
        # logging.warning("quant_file_dir is None, skipping saving quant data.")
        return
    file_path = Path(file_dir) / f"{file_name}.txt"
    try:
        tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(tensor_str)
        logging.info(f"Quantized data saved successfully at: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save quantized data: {e}")
