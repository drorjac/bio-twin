import numpy as np
import pytest

from biotwin.data.dataset import Cohort
from biotwin.data.simulate import patient_names


@pytest.fixture
def toy_cohort():
    """Tiny fake cohort (no simulator): 30 patients x 2 days, CGM = index so leakage is visible."""
    names = np.array(patient_names())
    P, T = len(names), 2 * 288
    rng = np.random.default_rng(0)
    cgm = np.tile(np.arange(T, dtype=np.float32), (P, 1)) + np.arange(P)[:, None] * 10_000
    return Cohort(names, cgm, cgm.copy(), rng.random((P, T)).astype(np.float32),
                  rng.random((P, T)).astype(np.float32), np.full(P, 0.5, np.float32))
