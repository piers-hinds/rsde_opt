import torch
import pytest
from rsde_opt import MomentumParticleSystem

DIM = 3
NUM_PART = 5
ALPHA = 1.0
BETA_VAL = 0.4
SIGMA_VAL = 0.05
GAMMA_VAL = 0.7
STEP_SIZE = 0.1


def sphere(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum(dim=-1)


def init_state(n: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(n, DIM)


def noop_proj(x: torch.Tensor):
    return None


@pytest.fixture
def mps():
    return MomentumParticleSystem(
        objective=sphere,
        initial_state=init_state,
        alpha=ALPHA,
        beta=lambda t: BETA_VAL,
        sigma=lambda t: SIGMA_VAL,
        gamma=GAMMA_VAL,
        dim=DIM,
        num_particles=NUM_PART,
        step_size=STEP_SIZE,
        projection=noop_proj,
        device="cpu",
    )


def test_momentum_initial_setup(mps):
    assert mps.state.shape == (NUM_PART, DIM)
    assert mps.velocity.shape == (NUM_PART, DIM)
    assert mps.h.item() == pytest.approx(STEP_SIZE)
    assert (mps.velocity == 0).all()


def test_momentum_single_step_shapes_and_time(mps):
    normals = torch.randn_like(mps.state)
    t0 = mps.t.clone()
    state_out, x_bar_out = mps.step(normals)
    assert mps.t.item() == pytest.approx(t0.item() + STEP_SIZE)
    assert state_out.shape == (NUM_PART, DIM)
    assert x_bar_out.shape == (DIM,)


def test_momentum_gamma_zero_reduces_to_original(mps):
    mps.gamma = lambda t: 0.0
    normals = torch.zeros_like(mps.state)
    x_bar = mps.consensus()
    delta = (mps.state - x_bar).clone()
    mps.step(normals)
    expected_velocity = -BETA_VAL * STEP_SIZE * delta
    assert torch.allclose(mps.velocity, expected_velocity)


def test_momentum_projection_functional_and_inplace():
    clamp_val = 0.3

    def clamp_proj(x: torch.Tensor):
        return x.clamp_(-clamp_val, clamp_val)

    sys = MomentumParticleSystem(
        objective=sphere,
        initial_state=init_state,
        alpha=ALPHA,
        beta=lambda t: BETA_VAL,
        sigma=lambda t: SIGMA_VAL,
        gamma=GAMMA_VAL,
        dim=DIM,
        num_particles=NUM_PART,
        step_size=STEP_SIZE,
        projection=clamp_proj,
    )

    normals = torch.randn_like(sys.state)
    sys.step(normals)
    assert sys.state.max() <= clamp_val + 1e-6
    assert sys.state.min() >= -clamp_val - 1e-6


def test_momentum_reset_clears_velocity(mps):
    normals = torch.randn_like(mps.state)
    mps.step(normals)
    assert (mps.velocity != 0).any()
    mps.reset()
    assert (mps.velocity == 0).all()


def test_momentum_shape_guard():
    with pytest.raises(ValueError):
        MomentumParticleSystem(
            objective=sphere,
            initial_state=lambda n: torch.zeros(n),
            alpha=ALPHA,
            beta=lambda t: BETA_VAL,
            sigma=lambda t: SIGMA_VAL,
            gamma=GAMMA_VAL,
            dim=DIM,
            num_particles=NUM_PART,
            step_size=STEP_SIZE,
        )
