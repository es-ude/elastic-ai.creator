def dequantizeProcess(x_q: int, scale: float, zero_point: int) -> float:
    "compatiable with symmetric/asymmetric, signed/unsignd de-quantization"
    return scale * (x_q - zero_point)
