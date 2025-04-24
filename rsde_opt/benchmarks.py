import torch
import numpy as np


def rastrigin_function(x: torch.Tensor) -> torch.Tensor:
    """
    Rastrigin function.

    Args:
    x: Input tensor of shape (n, d) where n is the number of points and d is the dimension.

    Returns:
    torch.Tensor: Output tensor of shape (n,) containing the Rastrigin function values for each input point.
    """
    d = x.shape[1]
    return 10 * d + torch.sum(x ** 2 - 10 * torch.cos(2 * np.pi * x), dim=1)


def ackley_function(x: torch.Tensor) -> torch.Tensor:
    """
    Ackley function, translated so the global minimum is at (2, 2)

    Args:
    x: Input tensor of shape (n, 2), where n is the number of points.
                      The tensor should contain 2-dimensional points.

    Returns:
    torch.Tensor: Output tensor of shape (n,) containing the Ackley function values
                  for each input point.
    """
    x_part, y_part = x[:, 0] - 2, x[:, 1] - 2
    a = 20
    b = 0.2
    c = 2 * torch.pi
    term1 = -a * torch.exp(-b * torch.sqrt(0.5 * (x_part ** 2 + y_part ** 2)))
    term2 = -torch.exp(0.5 * (torch.cos(c * x_part) + torch.cos(c * y_part)))
    ackley_value = term1 + term2 + a + torch.exp(torch.tensor(1.0))
    return ackley_value


def townsend_function(x: torch.Tensor) -> torch.Tensor:
    """
    Townsend function

    Args:
        x: Input tensor of shape (n, 2), where n is the number of points.
            The tensor should contain 2-dimensional points.

    Returns:
    Output tensor of shape (n,) containing the Townsend function values for each input point.

    """
    xs = x[:, 0]
    ys = x[:, 1]
    return - (torch.cos((xs - 0.1) * ys)) ** 2 - xs * torch.sin(3 * xs + ys)


def rosenbrock_function(x: torch.Tensor) -> torch.Tensor:
    """
    Rosenbrock function in 2 dimensions

    Args:
        x: Input tensor of shape (n, 2), where n in the number of points.

    Returns:
    Output tensor of shpae (n, ) containing the Rosenbrock function values for each input point.

    """
    xs, ys = x[:, 0], x[:, 1]
    return (1 - xs) ** 2 + 100 * (ys - xs ** 2) ** 2


def eggholder_function(x: torch.Tensor) -> torch.Tensor:
    """
    Eggholder function.

    Args:
        x (torch.Tensor): Input tensor of shape (n, 2), where each row is [x_i, y_i].

    Returns:
        torch.Tensor: Output tensor of shape (n,) containing the Eggholder function
                      values for each input point.
    """
    if x.dim() != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected input of shape (n, 2), got {tuple(x.shape)}")

    x1 = x[:, 0]
    x2 = x[:, 1]

    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x1 / 2 + (x2 + 47))))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))

    return term1 + term2
