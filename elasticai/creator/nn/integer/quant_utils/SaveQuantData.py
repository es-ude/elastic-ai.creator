from pathlib import Path


def save_quant_data(tensor, file_dir: Path, file_name: str):
    file_path = file_dir / f"{file_name}.txt"
    tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
    file_path.write_text(tensor_str)
