import torch
from torch import nn



def _prepare_ridge_targets(targets: torch.Tensor, num_outputs: int) -> torch.Tensor:
    if targets.dim() == 1 and num_outputs > 1 and not torch.is_floating_point(targets):
        return torch.nn.functional.one_hot(targets.long(), num_classes=num_outputs).float()

    targets = targets.float()
    if num_outputs == 1:
        return targets.reshape(-1, 1)

    return targets.reshape(targets.size(0), num_outputs)


def _construct_ridge_readout(num_features: int, num_outputs: int) -> nn.Linear:
    readout = nn.Linear(num_features, num_outputs)
    for param in readout.parameters():
        param.requires_grad = False
    return readout


def _fit_ridge_readout(
    readout: nn.Linear,
    features: torch.Tensor,
    targets: torch.Tensor,
    num_outputs: int,
    ridge_alpha: float = 1.0,
) -> None:
    if features.dim() != 2:
        raise ValueError("Ridge features must have shape (num_samples, num_features).")
    if features.size(0) != targets.size(0):
        raise ValueError("Ridge features and targets must contain the same number of samples.")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative.")

    solve_device = features.device
    x = features.detach().to(device=solve_device, dtype=torch.float64)
    y = _prepare_ridge_targets(targets.detach().to(solve_device), num_outputs).to(torch.float64)
    ones = torch.ones(x.size(0), 1, device=solve_device, dtype=x.dtype)
    x_aug = torch.cat([x, ones], dim=1)

    regularizer = ridge_alpha * torch.eye(x_aug.size(1), device=solve_device, dtype=x.dtype)
    regularizer[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + regularizer
    rhs = x_aug.T @ y

    try:
        solution = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        solution = torch.linalg.lstsq(lhs, rhs).solution

    with torch.no_grad():
        readout.weight.copy_(solution[:-1].T.to(device=readout.weight.device, dtype=readout.weight.dtype))
        readout.bias.copy_(solution[-1].to(device=readout.bias.device, dtype=readout.bias.dtype))
