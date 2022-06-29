"""Activation functions for Transducer."""

from distutils.version import LooseVersion

import torch


def get_activation(
    activation_type: str,
    ftswish_threshold: float = -0.2,
    ftswish_mean_shift: float = 0.0,
    hardtanh_min_val: int = -1.0,
    hardtanh_max_val: int = 1.0,
    leakyrelu_neg_slope: float = 0.01,
    smish_alpha: float = 1.0,
    smish_beta: float = 1.0,
    softplus_beta: float = 1.0,
    softplus_threshold: int = 20,
    swish_beta: float = 1.0,
) -> torch.nn.Module:
    """Return activation function.

    Args:
        activation_type: Activation type.
        ftswish_threshold: Threshold value for FTSwish activation formulation.
        ftswish_mean_shift: Mean shifting value for FTSwish activation formulation.
        hardtanh_min_val: Minimum value of the linear region range for HardTanh.
        hardtanh_max_val: Maximum value of the linear region range for HardTanh.
        leakyrelu_neg_slope: Negative slope value for LeakyReLU activation formulation.
        smish_alpha: Alpha value for Smish activation fomulation.
        smish_beta: Beta value for Smish activation formulation.
        softplus_beta: Beta value for softplus activation formulation in Mish.
        softplus_threshold: Values above this revert to a linear function in Mish.
        swish_beta: Beta value for Swish variant formulation.

    Returns:
        : Activation function.

    """
    torch_version = LooseVersion(torch.__version__)

    activations = {
        "ftswish": (
            FTSwish,
            {"threshold": ftswish_threshold, "mean_shift": ftswish_mean_shift},
        ),
        "hardtanh": (
            torch.nn.Hardtanh,
            {"min_val": hardtanh_min_val, "max_val": hardtanh_max_val},
        ),
        "leaky_relu": (torch.nn.LeakyReLU, {"negative_slope": leakyrelu_neg_slope}),
        "mish": (
            Mish,
            {
                "softplus_beta": softplus_beta,
                "softplus_threshold": softplus_threshold,
                "use_builtin": torch_version >= LooseVersion("1.9"),
            },
        ),
        "relu": (torch.nn.ReLU, {}),
        "selu": (torch.nn.SELU, {}),
        "smish": (Smish, {"alpha": smish_alpha, "beta": smish_beta}),
        "swish": (
            Swish,
            {"beta": swish_beta, "use_builtin": torch_version >= LooseVersion("1.8")},
        ),
        "tanh": (torch.nn.Tanh, {}),
    }

    act_func, act_args = activations[activation_type]

    return act_func(**act_args)


class FTSwish(torch.nn.Module):
    """Flatten-T Swish activation definition.

    Reference: https://arxiv.org/abs/1812.06247

    Args:
        threshold: Threshold value for FTSwish activation formulation.
        mean_shift: Mean shifting value for FTSwish activation formulation.

    """

    def __init__(self, threshold: float = -0.2, mean_shift: float = 0):
        super().__init__()

        self.threshold = threshold
        self.mean_shift = mean_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        x = (x * torch.sigmoid(x)) + self.threshold
        x = torch.where(x >= 0, x, torch.tensor([self.threshold], device=x.device))

        if self.mean_shift != 0:
            x.sub_(self.mean_shift)

        return x


class Mish(torch.nn.Module):
    """Mish activation definition.

    Reference: https://arxiv.org/abs/1908.08681.

    Args:
        softplus_beta: Beta value for softplus activation formulation.
        softplus_threshold: Values above this revert to a linear function.
        use_builtin: Whether to use PyTorch activation function if available.

    """

    def __init__(
        self,
        softplus_beta: float = 1.0,
        softplus_threshold: int = 20,
        use_builtin: bool = False,
    ):
        super().__init__()

        if use_builtin:
            self.mish = torch.nn.Mish()
        else:
            self.tanh = torch.nn.Tanh()
            self.softplus = torch.nn.Softplus(
                beta=softplus_beta, threshold=softplus_threshold
            )

            self.mish = lambda x: x * self.tanh(self.softplus(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        return self.mish(x)


class Smish(torch.nn.Module):
    """Smish activation definition.

    Reference: https://www.mdpi.com/2079-9292/11/4/540/htm.

    Args:
        alpha: Alpha value for Smish activation fomulation.
        beta: Beta value for Smish activation formulation.

    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()

        self.tanh = torch.nn.Tanh()

        self.alpha = alpha
        self.beta = beta

        self.smish = lambda x: (self.alpha * x) * self.tanh(
            torch.log(1 + torch.sigmoid((self.beta * x)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        return self.smish(x)


class Swish(torch.nn.Module):
    """Swish activation definition.

    References:
        https://arxiv.org/abs/2108.12943 / https://arxiv.org/abs/1710.05941v1.
        E-variant: https://arxiv.org/abs/1801.07145.

    Args:
        beta: Beta parameter for E-Swish variant.
        use_builtin: Whether to use PyTorch function if available.

    """

    def __init__(self, beta: float = 1.0, use_builtin: bool = False):
        super().__init__()

        if use_builtin:
            self.swish = torch.nn.SiLU()
        else:
            self.beta = beta

            self.swish = lambda x: (self.beta * x) * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation."""
        return self.swish(x)
