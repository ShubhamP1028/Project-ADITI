"""
PlantDiseaseSNN architecture — must match training code exactly.
Copied from BPTT_snn_cnn_izhikevich.py so app.py can reconstruct
the model before loading state_dict weights.
"""

import torch
import torch.nn as nn


# ── SuperSpike surrogate gradient ─────────────────────────────────────────
class SuperSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v: torch.Tensor, v_th: float = 30.0) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        return (v >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        surrogate = 1.0 / (1.0 + (v - ctx.v_th).abs()) ** 2
        return grad_output * surrogate, None


# ── Izhikevich parameter / state containers ───────────────────────────────
class IzhikevichParameters:
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0, v_th=30.0, dt=0.5, name=""):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.v_th = v_th
        self.dt   = dt
        self.name = name


class IzhikevichState:
    def __init__(self, v: torch.Tensor, u: torch.Tensor):
        self.v = v
        self.u = u

    def detach(self):
        return IzhikevichState(v=self.v.detach(), u=self.u.detach())


def izhikevich_step(input_current, state, params):
    v, u = state.v, state.u
    dt   = params.dt

    dv = (0.04 * v * v + 5.0 * v + 140.0 - u + input_current) * dt
    du = params.a * (params.b * v - u) * dt

    v_new = torch.clamp(v + dv, -100.0, 60.0)
    u_new = torch.clamp(u + du, -100.0, 100.0)

    spikes = SuperSpike.apply(v_new, params.v_th)
    v_out  = v_new * (1.0 - spikes) + params.c * spikes
    u_out  = u_new + params.d * spikes

    return spikes, IzhikevichState(v=v_out, u=u_out)


# ── Izhikevich SNN layer ──────────────────────────────────────────────────
class IzhikevichSNNLayer(nn.Module):
    def __init__(self, in_features, out_features, params, T=8):
        super().__init__()
        self.out_features = out_features
        self.params       = params
        self.T            = T

        self.linear      = nn.Linear(in_features, out_features, bias=True)
        self.layer_norm  = nn.LayerNorm(out_features)
        self.current_mag = nn.Parameter(torch.tensor(12.0))

        nn.init.kaiming_normal_(self.linear.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.linear.bias)

    def _init_state(self, batch, dev, dtype):
        v = torch.full((batch, self.out_features), -65.0, device=dev, dtype=dtype)
        u = torch.full((batch, self.out_features), self.params.b * -65.0, device=dev, dtype=dtype)
        return IzhikevichState(v=v, u=u)

    def forward(self, x):
        B     = x.size(0)
        raw   = self.layer_norm(self.linear(x))
        I_inj = torch.tanh(raw) * self.current_mag.abs()
        state = self._init_state(B, x.device, x.dtype)
        acc   = torch.zeros(B, self.out_features, device=x.device, dtype=x.dtype)

        for _ in range(self.T):
            I_t        = I_inj + 0.3 * torch.randn_like(I_inj)
            spk, state = izhikevich_step(I_t, state, self.params)
            acc       += spk
            state      = state.detach()

        return acc / self.T


# ── CNN backbone ──────────────────────────────────────────────────────────
class CNNBackbone(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin,  cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.GELU(),
            )

        self.stage1 = nn.Sequential(_block(3,   32),  nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(_block(32,  64),  nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(_block(64,  128), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(_block(128, 256), nn.AdaptiveAvgPool2d(4))

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.projector(x)


# ── Full model ────────────────────────────────────────────────────────────
class PlantDiseaseSNN(nn.Module):
    def __init__(self, neuron_params, num_classes, feature_dim=512, snn_hidden=256, T=8, name=""):
        super().__init__()
        self.name          = name
        self.neuron_params = neuron_params

        self.cnn  = CNNBackbone(feature_dim=feature_dim)
        self.snn1 = IzhikevichSNNLayer(feature_dim,       snn_hidden,      neuron_params, T)
        self.snn2 = IzhikevichSNNLayer(snn_hidden,        snn_hidden // 2, neuron_params, T)
        self.skip = nn.Sequential(
            nn.Linear(feature_dim, snn_hidden // 2, bias=False),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(snn_hidden // 2, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats   = self.cnn(x)
        snn_out = self.snn2(self.snn1(feats))
        gated   = snn_out + self.skip(feats)
        return self.classifier(gated)


# ── Regular Spiking params (used for regular_spiking_model.pt) ────────────
RS_PARAMS = IzhikevichParameters(a=0.02, b=0.20, c=-65.0, d=8.0, name="RS")
