"""Training objectives for spatial, spectral, and uncertainty fidelity."""

from .combined_loss import CombinedLoss
from .heteroscedastic_loss import heteroscedastic_l1_loss
from .mtf_psf_loss import MTFPSFLoss
from .spectral_consistency_loss import SpectralConsistencyLoss, spectral_angle_mapper

__all__ = [
    "CombinedLoss",
    "heteroscedastic_l1_loss",
    "MTFPSFLoss",
    "SpectralConsistencyLoss",
    "spectral_angle_mapper",
]

