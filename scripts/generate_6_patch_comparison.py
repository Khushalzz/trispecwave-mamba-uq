import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from models.trispecwave_mamba_uq import TriSpecWaveMambaUQ
from models.rgbn_unet_adapter import RGBNUNetAdapter

def _load_mamba(ckpt_path, cfg, device):
    model = TriSpecWaveMambaUQ(
        feature_dim=cfg["feature_dim"], spectral_dim=cfg["spectral_dim"],
        naf_depth=cfg["naf_depth"], spatial_depth=cfg["spatial_depth"],
        band_wavelengths_nm=cfg["band_wavelengths_nm"], band_gsd_m=cfg["band_gsd_m"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def generate_six_comparisons():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 6-patch + Pune evaluation on {device}...")

    with open("configs/sen2neon_6m.yaml") as f:
        cfg = yaml.safe_load(f)

    # Load models
    print("Loading old 6M checkpoint (sen2neon_6m_phase1/best.pt)...")
    model_old = _load_mamba("/mnt/pune/checkpoints/sen2neon_6m_phase1/best.pt", cfg, device)

    print("Loading DINO-L Phase 1 Final checkpoint (20261001T173510Z_final.pt)...")
    model_dino = _load_mamba("/mnt/pune/checkpoints/trispec_dino_l_sen2neon_phase1/20261001T173510Z_final.pt", cfg, device)

    print("Loading U-Net Adapter...")
    unet = RGBNUNetAdapter(in_channels=24, out_channels=4).to(device)
    unet.load_state_dict(torch.load("/mnt/pune/checkpoints/sen2neon_unet_adapter/epoch_3.pt", map_location=device, weights_only=True))
    unet.eval()

    # ── SEN2NEON VAL PATCHES ──────────────────────────────────────────────────
    val_dir = "/mnt/pune/sen2neon_patches_16x/val"
    rgb_bands = [3, 2, 1]

    candidate_patches = []
    for chunk_file in ["chunk_0001.pt", "chunk_0002.pt"]:
        path = os.path.join(val_dir, chunk_file)
        if not os.path.exists(path):
            continue
        data = torch.load(path, weights_only=True)
        lr_all = (data["lr"].float() / 10000.0).clamp(0, 1)
        hr_all = (data["hr"].float() / 10000.0).clamp(0, 1)
        for idx in range(0, len(hr_all), 15):
            hr_patch = hr_all[idx, rgb_bands]
            std_val = hr_patch.std().item()
            mean_val = hr_patch.mean().item()
            if mean_val > 0.03 and std_val > 0.04:
                candidate_patches.append({
                    "chunk": chunk_file, "idx": idx,
                    "lr": lr_all[idx:idx+1], "hr": hr_all[idx:idx+1],
                    "std": std_val, "is_pune": False
                })
        del data, lr_all, hr_all

    candidate_patches.sort(key=lambda x: x["std"], reverse=True)
    print(f"Found {len(candidate_patches)} SEN2NEON candidate patches.")
    step_size = max(1, len(candidate_patches) // 8)
    selected = [candidate_patches[i * step_size] for i in range(6)]

    # ── PUNE ROW (unpaired) ───────────────────────────────────────────────────
    cache_dir = "/mnt/pune/teacher_cache"
    pune_lr = None
    if os.path.exists(cache_dir):
        files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if files:
            patch_file = files[min(42, len(files) - 1)]
            data = torch.load(patch_file, map_location="cpu", weights_only=True)
            pune_lr = data["input_patch"].unsqueeze(0).float()
            print(f"Loaded Pune patch from {patch_file}")

    NUM_ROWS = 7  # 6 val + 1 pune
    NUM_COLS = 6
    fig, axes = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(24, 28))
    fig.patch.set_facecolor("#0d0d0d")

    col_titles = [
        "1. Raw LR Input\n(64×64 Pixelated)",
        "2. Bicubic Baseline\n(256×256)",
        "3. 6M Mamba (Old Baseline)\n(sen2neon_6m_phase1)",
        "4. DINO-L Phase 1 (Final)\n(20261001T173510Z)",
        "5. 6M + U-Net Detailer\n(Our Pipeline)",
        "6. HR Ground Truth\n(AVIRIS-NG 2.5m)"
    ]
    col_colors = ["white", "white", "white", "#f59e0b", "#4ade80", "white"]

    for col_idx, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=10, color=color)

    def run_inference(lr_tensor):
        with torch.no_grad():
            x = lr_tensor.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_old  = model_old(x)
                out_dino = model_dino(x)
            pred_old  = (out_old[0]  if isinstance(out_old,  tuple) else out_old).clamp(0, 1)
            pred_dino = (out_dino[0] if isinstance(out_dino, tuple) else out_dino).clamp(0, 1)
            bicubic   = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                unet_out = unet(pred_old, bicubic)
            unet_pred = unet_out.clamp(0, 1)
        return pred_old, pred_dino, bicubic, unet_pred

    def norm_by(ref_rgb, x):
        p2, p98 = np.percentile(ref_rgb, 2), np.percentile(ref_rgb, 98)
        return np.clip((x - p2) / max(p98 - p2, 1e-4), 0, 1)

    def render_row(row_idx, lr, hr, row_label, is_pune=False):
        pred_old, pred_dino, bicubic, unet_pred = run_inference(lr)

        raw_lr_vis  = F.interpolate(lr[0:1, rgb_bands], size=(256, 256), mode="nearest")[0].cpu().numpy().transpose(1, 2, 0)
        bic_rgb     = bicubic[0, rgb_bands].cpu().numpy().transpose(1, 2, 0)
        old_rgb     = pred_old[0, rgb_bands].cpu().float().numpy().transpose(1, 2, 0)
        dino_rgb    = pred_dino[0, rgb_bands].cpu().float().numpy().transpose(1, 2, 0)
        unet_rgb    = unet_pred[0, [2, 1, 0]].cpu().float().numpy().transpose(1, 2, 0)
        hr_rgb      = hr[0, rgb_bands].cpu().numpy().transpose(1, 2, 0) if hr is not None else None

        ref = hr_rgb if hr_rgb is not None else bic_rgb

        if not is_pune:
            psnr_bic  = calc_psnr(hr_rgb, bic_rgb,  data_range=1.0)
            ssim_bic  = calc_ssim(hr_rgb, bic_rgb,  channel_axis=2, data_range=1.0)
            psnr_old  = calc_psnr(hr_rgb, old_rgb,  data_range=1.0)
            ssim_old  = calc_ssim(hr_rgb, old_rgb,  channel_axis=2, data_range=1.0)
            psnr_dino = calc_psnr(hr_rgb, dino_rgb, data_range=1.0)
            ssim_dino = calc_ssim(hr_rgb, dino_rgb, channel_axis=2, data_range=1.0)
            psnr_unet = calc_psnr(hr_rgb, unet_rgb, data_range=1.0)
            ssim_unet = calc_ssim(hr_rgb, unet_rgb, channel_axis=2, data_range=1.0)
            labels = [
                None,
                f"PSNR: {psnr_bic:.2f} dB | SSIM: {ssim_bic:.4f}",
                f"PSNR: {psnr_old:.2f} dB | SSIM: {ssim_old:.4f}",
                f"PSNR: {psnr_dino:.2f} dB | SSIM: {ssim_dino:.4f}",
                f"PSNR: {psnr_unet:.2f} dB | SSIM: {ssim_unet:.4f}",
                "Target (Ground Truth)"
            ]
        else:
            labels = [None, "No GT (Pune)", "No GT (Pune)", "No GT (Pune)", "No GT (Pune)", "No Ground Truth"]

        lc = ["white", "white", "white", "#f59e0b", "#4ade80", "#888888"]
        imgs = [norm_by(ref, raw_lr_vis), norm_by(ref, bic_rgb), norm_by(ref, old_rgb),
                norm_by(ref, dino_rgb), norm_by(ref, unet_rgb),
                norm_by(ref, hr_rgb) if hr_rgb is not None else np.zeros((256, 256, 3))]

        for col_idx, (img, lbl, color) in enumerate(zip(imgs, labels, lc)):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap="gray" if img.max() == 0 else None)
            ax.set_facecolor("#0d0d0d")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")
            ax.set_xticks([])
            ax.set_yticks([])
            if lbl:
                ax.set_xlabel(lbl, fontsize=9, color=color,
                              fontweight="bold" if col_idx in [3, 4] else "normal")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight="bold", color="#ff6b6b" if is_pune else "white")

    # Render SEN2NEON rows
    for row_idx, item in enumerate(selected):
        render_row(
            row_idx,
            item["lr"], item["hr"],
            row_label=f"Patch #{row_idx+1}\n({item['chunk']}:{item['idx']})"
        )

    # Render Pune row
    if pune_lr is not None:
        render_row(6, pune_lr, None, row_label="Pune, India\n(Unpaired — No GT)", is_pune=True)
    else:
        for col_idx in range(NUM_COLS):
            axes[6, col_idx].text(0.5, 0.5, "Pune data\nnot found",
                                  ha="center", va="center", color="red", transform=axes[6, col_idx].transAxes)
            axes[6, col_idx].set_facecolor("#0d0d0d")
            axes[6, col_idx].set_xticks([])
            axes[6, col_idx].set_yticks([])

    plt.tight_layout(pad=0.4)
    out_file = "/mnt/pune/six_patch_dino_final_pune.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    generate_six_comparisons()
