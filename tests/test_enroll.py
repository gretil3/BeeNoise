import numpy as np
import pytest

from src.enroll import enroll_audio


@pytest.mark.parametrize("name", ["../../x", "a/b", "..", ".hidden", "", "x" * 65])
def test_unsafe_names_are_rejected_before_anything_is_written(name):
    with pytest.raises(ValueError, match="Name may only"):
        enroll_audio(name, np.zeros(16000, dtype=np.float32), 16000)
