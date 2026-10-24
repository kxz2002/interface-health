import numpy as np

from scripts.service_current_semantic_model import _cosine_distance, _robust_z


def test_cosine_distance_uses_service_specific_normal_reference():
    vectors = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    references = {
        "a": np.asarray([1, 0], dtype=np.float32),
        "b": np.asarray([0, 1], dtype=np.float32),
    }
    result = _cosine_distance(vectors, references, np.asarray(["a", "a", "b"]))
    np.testing.assert_allclose(result, [0.0, 1.0, 1 - 2**-0.5], atol=1e-6)


def test_robust_z_is_fitted_from_normal_reference_only():
    result = _robust_z(np.asarray([1.0, 3.0]), np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result, [-0.674491, 0.674491], atol=1e-5)
