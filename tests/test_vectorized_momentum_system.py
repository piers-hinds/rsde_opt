import torch
import pytest
from rsde_opt import VecMomentumParticleSystem


DIM           = 3
NUM_PART      = 4
NUM_EXP       = 2
STEP_SIZE     = 0.1
ALPHA         = 1.0
BETA_VAL      = 0.5
SIGMA_VAL     = 0.1
GAMMA_VAL     = 0.8


def sphere(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum(dim=-1)


def init_state(n: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, DIM)


def no_proj(x: torch.Tensor):
    return None 


@pytest.fixture
def system():
    return VecMomentumParticleSystem(
        objective=sphere,
        initial_state=init_state,
        alpha=ALPHA,
        beta=lambda t: BETA_VAL,
        sigma=lambda t: SIGMA_VAL,
        gamma=GAMMA_VAL,
        dim=DIM,
        num_particles=NUM_PART,
        num_experiments=NUM_EXP,
        step_size=STEP_SIZE,
        projection=no_proj,
        device="cpu",
    )


def test_vec_mom_initial_shapes(system):
    assert system.state.shape == (NUM_EXP, NUM_PART, DIM)
    assert system.velocity.shape == system.state.shape
    assert system.h.item() == pytest.approx(STEP_SIZE)


def test_vec_mom_time_advances_and_shapes_preserved(system):
    normals = torch.randn_like(system.state)
    old_t = system.t.clone()
    state_out, x_bar_out = system.step(normals)
    assert system.t.item() == pytest.approx(old_t.item() + STEP_SIZE)
    assert state_out.shape == (NUM_EXP, NUM_PART, DIM)
    assert x_bar_out.shape == (NUM_EXP, DIM)


def test_vec_mom_gamma_zero_equals_original(system):
    system.gamma = lambda t: 0.0
    normals = torch.zeros_like(system.state)
    x_bar = system.consensus().unsqueeze(1)
    delta = (system.state - x_bar).clone()
    system.step(normals)
    expected = delta * (-BETA_VAL * STEP_SIZE)
    assert torch.allclose(system.velocity, expected)


def test_vec_mom_projection_called(system):
    def clamp_proj(x: torch.Tensor):
        return x.clamp_(-0.5, 0.5)

    system.projection = clamp_proj
    normals = torch.randn_like(system.state)
    system.step(normals)
    assert system.state.max() <= 0.5 + 1e-6
    assert system.state.min() >= -0.5 - 1e-6


def test_vec_mom_bad_init_shape():
    with pytest.raises(ValueError):
        VecMomentumParticleSystem(
            objective=sphere,
            initial_state=lambda n: torch.zeros(n),
            alpha=ALPHA,
            beta=lambda t: BETA_VAL,
            sigma=lambda t: SIGMA_VAL,
            gamma=GAMMA_VAL,
            dim=DIM,
            num_particles=NUM_PART,
            num_experiments=NUM_EXP,
            step_size=STEP_SIZE,
        )