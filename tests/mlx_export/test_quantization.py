import mlx.core as mx
import numpy as np

from boltz_mlx_export.quantization import quantize_affine_int8


def test_short_linear_is_zero_padded() -> None:
    """A narrow learned projection remains eligible for int8 export."""
    weight = np.arange(8, dtype=np.float16).reshape(8, 1)

    quantized = quantize_affine_int8(weight)

    assert quantized.logical_shape == (8, 1)
    assert quantized.physical_shape == (8, 64)
    assert quantized.weight.dtype == np.uint32
    restored = mx.dequantize(
        mx.array(quantized.weight),
        mx.array(quantized.scales),
        mx.array(quantized.biases),
        group_size=64,
        bits=8,
        mode="affine",
    )
    restored_values = np.asarray(restored)
    np.testing.assert_allclose(restored_values[:, :1], weight, atol=0.03, rtol=0)
    np.testing.assert_allclose(restored_values[:, 1:], 0, atol=0.03, rtol=0)


def test_aligned_linear_is_not_padded() -> None:
    """An already aligned matrix retains its physical shape."""
    weight = np.zeros((2, 64), dtype=np.float16)

    quantized = quantize_affine_int8(weight)

    assert quantized.logical_shape == quantized.physical_shape == (2, 64)


def test_quantization_rejects_non_matrix() -> None:
    """Affine matrix export requires exactly two dimensions."""
    with np.testing.assert_raises_regex(ValueError, "two-dimensional"):
        quantize_affine_int8(np.zeros((64,), dtype=np.float16))
