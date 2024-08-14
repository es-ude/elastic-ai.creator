from pathlib import Path


def save_quant_data(tensor, file_dir: Path, file_name: str):
    file_path = file_dir / f"{file_name}.txt"
    tensor_str = "\n".join(map(str, tensor.flatten().tolist()))
    file_path.write_text(tensor_str)


# def save_quant_data(q_data, q_data_name, model_for_hw_path, layer_name):
#     q_data_numpy = q_data.int().cpu().numpy()
#     q_data_file_path = os.path.join(
#         model_for_hw_path, f"{layer_name}_{q_data_name}.txt"
#     )
#     if q_data_file_path is not None:
#         with open(q_data_file_path, "a") as f:
#             np.savetxt(f, q_data_numpy.reshape(-1, 1), fmt="%d")
#     else:
#         raise ValueError("q_data_file_path is None")
