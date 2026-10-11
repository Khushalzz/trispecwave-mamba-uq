"""Unit and shape verification tests for OpenSR-SRGAN to TriSpecWave-Mamba-UQ distillation.

Verifies:
1. No B10 is present in band definition or groups.
2. Band order strictly matches: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12.
3. Teacher input has 4 bands corresponding to B2, B3, B4, B8.
4. Student model input accepts [B, 12, 64, 64].
5. Teacher output is 4x upsampled: [B, 4, 256, 256].
6. Student output is 4x upsampled: [B, 12, 256, 256].
7. MTF/PSF degradation recovers exact [B, 12, 64, 64] shape.
8. Cached teacher files can be loaded and conform to expected structure.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import yaml

from losses.mtf_psf_loss import MTFPSFLoss
from losses.spectral_consistency_loss import spectral_angle_mapper
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ

EXPECTED_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
TEACHER_BANDS = [1, 2, 3, 7]  # B2, B3, B4, B8


def test_band_configuration():
    """Verify that B10 is strictly absent and band definitions are exact."""
    config_path = Path("configs/distill_pune.yaml")
    if not config_path.exists():
        config_path = Path("/root/srm_repo/configs/distill_pune.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    band_names = cfg.get("band_names", [])
    assert "B10" not in band_names, "B10 (cirrus) must never be included!"
    assert len(band_names) == 12, f"Expected 12 bands, got {len(band_names)}"
    assert band_names == EXPECTED_BANDS, f"Band order mismatch: {band_names} != {EXPECTED_BANDS}"

    groups = cfg.get("groups", {})
    assert groups["native_10m"] == [1, 2, 3, 7], "native_10m must correspond to B2, B3, B4, B8"
    assert groups["native_20m"] == [4, 5, 6, 8, 10, 11], "native_20m must correspond to B5, B6, B7, B8A, B11, B12"
    assert groups["native_60m"] == [0, 9], "native_60m must correspond to B1, B9"


def test_teacher_input_and_output_shapes():
    """Verify teacher input slice has 4 bands and predict_step output is 4x upsampled."""
    try:
        from opensr_srgan import load_inference_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        teacher = load_inference_model("RGB-NIR").to(device).eval()

        # Student input mock [1, 12, 64, 64]
        x_12b = torch.rand(1, 12, 64, 64, device=device)
        teacher_in = x_12b[:, TEACHER_BANDS]

        assert teacher_in.shape == (1, 4, 64, 64), f"Teacher input shape mismatch: {teacher_in.shape}"

        with torch.no_grad():
            teacher_out = teacher.predict_step(teacher_in)

        assert teacher_out.shape == (1, 4, 256, 256), f"Teacher output shape mismatch: {teacher_out.shape}"
    except ImportError:
        pytest.skip("opensr_srgan not installed in test environment")


def test_student_forward_shapes():
    """Verify student model initializes with feature_dim=32 and produces [B, 12, 256, 256]."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = TriSpecWaveMambaUQ(
        in_channels=12,
        out_channels=12,
        feature_dim=32,
        spectral_dim=32,
        upsample_stages=2,
    ).to(device).eval()

    x = torch.rand(2, 12, 64, 64, device=device)
    with torch.no_grad():
        hr, logvar = student(x)

    assert hr.shape == (2, 12, 256, 256), f"Student HR shape mismatch: {hr.shape}"
    assert logvar.shape == (2, 12, 256, 256), f"Student logvar shape mismatch: {logvar.shape}"


def test_mtf_sensor_degradation_shape():
    """Verify MTF degradation returns exact input resolution [B, 12, 64, 64]."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtf = MTFPSFLoss(scale=4, kernel_size=5, sigma=1.2).to(device)

    pred_hr = torch.rand(2, 12, 256, 256, device=device)
    degraded = mtf.degrade(pred_hr, output_size=(64, 64))

    assert degraded.shape == (2, 12, 64, 64), f"MTF degraded shape mismatch: {degraded.shape}"


def test_distillation_loss_computation():
    """Verify all loss terms calculate without errors and produce finite scalars."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtf = MTFPSFLoss(scale=4, kernel_size=5, sigma=1.2).to(device)

    x = torch.rand(1, 12, 64, 64, device=device)
    student_hr = torch.rand(1, 12, 256, 256, device=device)
    teacher_hr = torch.rand(1, 4, 256, 256, device=device)

    student_rgbn = student_hr[:, TEACHER_BANDS]
    l_teacher = F.l1_loss(student_rgbn, teacher_hr)

    degraded_12b = mtf.degrade(student_hr, output_size=x.shape[-2:])
    l_sensor = F.l1_loss(degraded_12b, x)

    l_sam = spectral_angle_mapper(student_rgbn, teacher_hr)

    total_loss = 1.0 * l_teacher + 0.25 * l_sensor + 0.05 * l_sam

    assert torch.isfinite(total_loss), "Distillation loss must be finite!"
    assert total_loss.item() > 0.0, "Distillation loss should be positive"


def test_cached_patch_format():
    """Verify structure and metadata of any existing cached teacher files."""
    cache_dir = Path("/mnt/pune/teacher_cache")
    if not cache_dir.exists():
        pytest.skip("Volume cache directory does not exist locally (run on Modal)")

    cached_files = list(cache_dir.glob("*.pt"))
    if not cached_files:
        pytest.skip("No cached target files found yet")

    sample_file = cached_files[0]
    data = torch.load(sample_file, map_location="cpu")

    assert "input_patch" in data, "Cached item missing input_patch"
    assert "teacher_output" in data, "Cached item missing teacher_output"
    assert data["input_patch"].shape == (12, 64, 64), f"Input patch shape wrong: {data['input_patch'].shape}"
    assert data["teacher_output"].shape == (4, 256, 256), f"Teacher output shape wrong: {data['teacher_output'].shape}"
    assert "patch_id" in data, "Cached item missing patch_id"
    assert "split" in data, "Cached item missing split"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
