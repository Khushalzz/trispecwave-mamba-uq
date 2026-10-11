"""Monte Carlo dropout epistemic uncertainty."""

from __future__ import annotations

import torch
from torch import nn


def _enable_dropout(module: nn.Module) -> None:
    if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, samples: int = 8) -> dict[str, torch.Tensor]:
    """Run stochastic passes and return mean, epistemic variance, and logvar."""

    if samples < 2:
        raise ValueError("At least two Monte Carlo samples are required")
    was_training = model.training
    model.eval()
    model.apply(_enable_dropout)
    means, logvars = [], []
    for _ in range(samples):
        output = model(x)
        if isinstance(output, dict):
            mean, logvar = output["mean"], output.get("logvar")
        else:
            mean, logvar = output[0], output[1] if isinstance(output, tuple) else None
        means.append(mean)
        if logvar is not None:
            logvars.append(logvar)
    stacked = torch.stack(means)
    if was_training:
        model.train()
    result = {
        "mean": stacked.mean(dim=0),
        "epistemic_variance": stacked.var(dim=0, unbiased=False),
    }
    if logvars:
        result["logvar"] = torch.stack(logvars).mean(dim=0)
        result["aleatoric_variance"] = result["logvar"].exp()
    result["total_variance"] = result["epistemic_variance"] + result.get("aleatoric_variance", 0.0)
    return result

