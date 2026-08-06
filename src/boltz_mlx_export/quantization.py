"""MLX affine-int8 conversion for PyTorch matrix parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

INT8_BITS = 8
INT8_GROUP_SIZE = 64
MATRIX_RANK = 2


@dataclass(frozen=True)
class QuantizedMatrix:
    """Packed affine matrix plus its logical and padded physical shapes."""

    weight: NDArray[np.uint32]
    scales: NDArray[np.floating]
    biases: NDArray[np.floating]
    logical_shape: tuple[int, int]
    physical_shape: tuple[int, int]


def quantize_affine_int8(
    weight: NDArray[np.floating],
    *,
    group_size: int = INT8_GROUP_SIZE,
) -> QuantizedMatrix:
    """Zero-pad and quantize one output-by-input matrix with MLX affine int8."""
    if weight.ndim != MATRIX_RANK:
        message = f"quantized weight must be two-dimensional, found rank {weight.ndim}"
        raise ValueError(message)
    output_width, logical_input_width = (int(value) for value in weight.shape)
    physical_input_width = (
        (logical_input_width + group_size - 1) // group_size
    ) * group_size
    padded = np.zeros((output_width, physical_input_width), dtype=np.float16)
    padded[:, :logical_input_width] = weight.astype(np.float16, copy=False)

    packed, scales, biases = mx.quantize(
        mx.array(padded),
        group_size=group_size,
        bits=INT8_BITS,
        mode="affine",
    )
    mx.eval(packed, scales, biases)
    return QuantizedMatrix(
        weight=np.asarray(packed),
        scales=np.asarray(scales),
        biases=np.asarray(biases),
        logical_shape=(output_width, logical_input_width),
        physical_shape=(output_width, physical_input_width),
    )
