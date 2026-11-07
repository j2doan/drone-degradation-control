"""Self-test for sensors/noise.py.

Run: python sensors/test_noise.py (or with pytest)
"""

import pathlib
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # drone/

import torch

from sensors.noise import ImuNoise, ImuNoiseCfg

HOVER_ACC = torch.tensor([[0.0, 0.0, 9.81]])


def clean(n=1):
    return HOVER_ACC.expand(n, 3).clone(), torch.zeros(n, 3)


def test_deterministic():
    a, b, c = ImuNoise(3), ImuNoise(3), ImuNoise(4)
    acc, gyro = clean()
    for _ in range(10):
        (aa, ag), (ba, bg), (ca, _) = (
            a.apply(acc, gyro),
            b.apply(acc, gyro),
            c.apply(acc, gyro),
        )
        assert torch.equal(aa, ba) and torch.equal(ag, bg)
        assert not torch.equal(aa, ca)
    print("deterministic: same seed -> identical sequence, other seed differs")


def test_inputs_untouched_and_bias_constant():
    noise = ImuNoise(0, num_envs=2)
    acc, gyro = clean(2)
    acc0, gyro0 = acc.clone(), gyro.clone()
    bias_a, bias_g = noise.accel_bias.clone(), noise.gyro_bias.clone()
    out_acc, out_gyro = noise.apply(acc, gyro)
    for _ in range(50):
        noise.apply(acc, gyro)
    assert torch.equal(acc, acc0) and torch.equal(gyro, gyro0)
    assert out_acc is not acc and out_gyro is not gyro
    assert torch.equal(noise.accel_bias, bias_a) and torch.equal(
        noise.gyro_bias, bias_g
    )
    assert bias_a.abs().sum() > 0 and bias_g.abs().sum() > 0
    print(
        "inputs untouched; bias constant within a flight "
        f"(accel bias env0={[round(v, 4) for v in bias_a[0].tolist()]})"
    )


def test_white_noise_std():
    cfg = ImuNoiseCfg()
    n, calls = 64, 200
    noise = ImuNoise(1, num_envs=n, cfg=cfg)
    acc, gyro = clean(n)
    res_a, res_g = [], []
    for _ in range(calls):
        na, ng = noise.apply(acc, gyro)
        res_a.append(na - acc - noise.accel_bias)
        res_g.append(ng - gyro - noise.gyro_bias)
    std_a, std_g = torch.stack(res_a).std().item(), torch.stack(res_g).std().item()
    assert abs(std_a / cfg.accel_noise_std - 1) < 0.05, std_a
    assert abs(std_g / cfg.gyro_noise_std - 1) < 0.05, std_g
    # the bias spread across envs is near its configured std too (loose: 64*3 samples)
    bstd = noise.accel_bias.std().item()
    assert abs(bstd / cfg.accel_bias_std - 1) < 0.3, bstd
    print(
        f"white noise std: accel {std_a:.4f} (cfg {cfg.accel_noise_std}), "
        f"gyro {std_g:.5f} (cfg {cfg.gyro_noise_std});"
        f" accel bias spread {bstd:.4f} (cfg {cfg.accel_bias_std})"
    )


def test_batched_and_reset():
    n = 5
    noise = ImuNoise(2, num_envs=n)
    acc, gyro = clean(n)
    na, ng = noise.apply(acc, gyro)
    assert na.shape == (n, 3) and ng.shape == (n, 3)
    bias_a, bias_g = noise.accel_bias.clone(), noise.gyro_bias.clone()
    noise.reset(torch.tensor([1, 3]))
    changed = (noise.accel_bias != bias_a).any(dim=-1) & (
        noise.gyro_bias != bias_g
    ).any(dim=-1)
    unchanged = (noise.accel_bias == bias_a).all(dim=-1) & (
        noise.gyro_bias == bias_g
    ).all(dim=-1)
    assert changed[[1, 3]].all() and unchanged[[0, 2, 4]].all(), (changed, unchanged)
    noise.reset()
    assert (noise.accel_bias != bias_a).any(dim=-1).all()
    print(
        f"batched N={n}: shapes ok; "
        "reset([1, 3]) changed only those biases; reset() changed all"
    )


def test_zero_cfg_is_identity():
    noise = ImuNoise(0, num_envs=2, cfg=ImuNoiseCfg(0.0, 0.0, 0.0, 0.0))
    acc, gyro = clean(2)
    na, ng = noise.apply(acc, gyro)
    assert torch.equal(na, acc) and torch.equal(ng, gyro)
    print("zero cfg: output equals input")


if __name__ == "__main__":
    test_deterministic()
    test_inputs_untouched_and_bias_constant()
    test_white_noise_std()
    test_batched_and_reset()
    test_zero_cfg_is_identity()
    print("ALL NOISE TESTS PASSED")
