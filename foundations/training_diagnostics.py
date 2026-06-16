import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        with torch.no_grad():
            out = x

            for layer in model:
                out = layer(out)

                if isinstance(layer, nn.Linear):
                    mean = round(out.mean().item(), 4)
                    std = round(out.std().item(), 4)

                    # Neuron is dead if output <= 0 for ALL samples
                    dead_neurons = (out <= 0).all(dim=0)
                    dead_fraction = round(
                        dead_neurons.float().mean().item(), 4
                    )

                    stats.append({
                        "mean": mean,
                        "std": std,
                        "dead_fraction": dead_fraction
                    })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        criterion = nn.MSELoss()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()

        stats = []

        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # 1. Dead neurons
        for stat in activation_stats:
            if stat["dead_fraction"] > 0.5:
                return "dead_neurons"

        # 2. Exploding gradients
        for stat in gradient_stats:
            if stat["norm"] > 1000:
                return "exploding_gradients"

        # 3. Vanishing gradients (last layer)
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation std checks
        for stat in activation_stats:
            if stat["std"] < 0.1:
                return "vanishing_gradients"
            if stat["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"
