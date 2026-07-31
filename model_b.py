"""
model_b.py — Model B: Quantum-Assisted PINN (QAPINN).

Same forward(x, t) -> u interface as model.py's classical PINN, so this
model is a drop-in replacement in train.py / evaluate.py / visualize.py —
per the project's controlled A/B design, this file is the ONLY thing that
should differ from Model A's pipeline (see report Section 5.1).

Architecture (default, INPUT_MODE='direct'):

    (x, t)
      │
      ▼
    quantum data-encoding + variational circuit   <- replaces Model A's
      │  (measures <Z> on each qubit)                 classical input layer
      ▼
    small classical linear head (n_qubits -> ... -> 1)
      │
      ▼
    u(x, t)

Three configurable pieces, each independently testable (see ablation_runner.py):

  - ENCODING: how (x, t) enters the circuit.
      'angle'             : encode once, before the first variational layer.
      'angle_reuploading' : re-encode (x, t) before EVERY variational layer
                             ("data re-uploading"). Per Schuld, Sweke &
                             Meyer (2021), repeating the encoding expands
                             the set of accessible frequencies Ω in the
                             circuit's output — this is the encoding this
                             project's central hypothesis is built around
                             (see report Section 4.2).
      'amplitude'         : (x, t) are lifted to a 2^n_qubits-dimensional
                             polynomial feature vector and loaded via
                             amplitude embedding (normalized state prep).

  - ENTANGLEMENT: 'linear' | 'circular' | 'full' CNOT pattern between
      variational layers.

  - CIRCUIT_DEPTH: number of variational (encode -> rotate -> entangle)
      layers.
"""
import math
import torch
import torch.nn as nn
import pennylane as qml


# ----------------------------------------------------------------------
# Circuit building blocks
# ----------------------------------------------------------------------

def _entangling_layer(n_qubits: int, pattern: str):
    if n_qubits < 2:
        return  # nothing to entangle with a single qubit
    if pattern == "linear":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif pattern == "circular":
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    elif pattern == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
    else:
        raise ValueError(f"Unknown entanglement pattern '{pattern}'. "
                          f"Choices: linear, circular, full.")


def _angle_encode(inputs, n_qubits: int):
    """
    Encodes the 2D input (x, t) across all qubits via RY + RZ rotations,
    alternating which variable each qubit carries. Using both RY and RZ
    (rather than a single rotation) gives the encoding generator a richer
    eigenvalue spectrum, which per the Fourier-series theorem in report
    Section 4.2 directly expands the set of accessible output frequencies.

    `inputs` has shape (..., 2) — supports batched execution.
    """
    x = inputs[..., 0]
    t = inputs[..., 1]
    for q in range(n_qubits):
        val = x if q % 2 == 0 else t
        qml.RY(math.pi * val, wires=q)
        qml.RZ(math.pi * val, wires=q)


def _amplitude_features(inputs, dim: int):
    """
    Lifts (x, t) into a `dim`-length polynomial feature vector
    [1, x, t, x^2, xt, t^2, ...] (monomials ordered by ascending total
    degree), used as the (pre-normalization) state to amplitude-encode.
    This is a documented design choice, not claimed to be optimal —
    amplitude encoding needs a length-2^n_qubits vector from a 2D input,
    and a low-degree polynomial expansion is the simplest defensible way
    to construct one without discarding information asymmetrically.
    """
    x, t = inputs[..., 0], inputs[..., 1]
    feats = []
    total_degree = 0
    while len(feats) < dim:
        for i in range(total_degree + 1):
            j = total_degree - i
            feats.append((x ** i) * (t ** j))
            if len(feats) == dim:
                break
        total_degree += 1
    return torch.stack(feats, dim=-1)


def build_qnode(n_qubits: int, depth: int, encoding: str, entanglement: str,
                 device_name: str = "default.qubit", diff_method: str = "backprop"):
    """
    Builds and returns (circuit, weight_shapes) for use with
    qml.qnn.TorchLayer. `circuit(inputs, weights)` maps a batch of (x, t)
    pairs, shape (N, 2), to a batch of n_qubits expectation values,
    shape (N, n_qubits).
    """
    dev = qml.device(device_name, wires=n_qubits)
    dim = 2 ** n_qubits

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        if encoding == "amplitude":
            feat = _amplitude_features(inputs, dim)
            qml.AmplitudeEmbedding(feat, wires=range(n_qubits), normalize=True)
            for l in range(depth):
                for q in range(n_qubits):
                    qml.Rot(*weights[l, q], wires=q)
                _entangling_layer(n_qubits, entanglement)

        elif encoding in ("angle", "angle_reuploading"):
            reupload = (encoding == "angle_reuploading")
            for l in range(depth):
                if l == 0 or reupload:
                    _angle_encode(inputs, n_qubits)
                for q in range(n_qubits):
                    qml.Rot(*weights[l, q], wires=q)
                _entangling_layer(n_qubits, entanglement)

        else:
            raise ValueError(f"Unknown encoding '{encoding}'. "
                              f"Choices: angle, angle_reuploading, amplitude.")

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    weight_shapes = {"weights": (depth, n_qubits, 3)}
    return circuit, weight_shapes


# ----------------------------------------------------------------------
# Hybrid quantum-classical PINN
# ----------------------------------------------------------------------

class QAPINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        circuit, weight_shapes = build_qnode(
            n_qubits=cfg.N_QUBITS, depth=cfg.CIRCUIT_DEPTH,
            encoding=cfg.ENCODING, entanglement=cfg.ENTANGLEMENT,
            device_name=cfg.QUANTUM_DEVICE, diff_method=cfg.DIFF_METHOD,
        )
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        if cfg.INPUT_MODE == "preprocessed":
            self.preproc = nn.Sequential(
                nn.Linear(2, cfg.CLASSICAL_PREPROCESS_DIM), nn.Tanh(),
                nn.Linear(cfg.CLASSICAL_PREPROCESS_DIM, 2), nn.Tanh(),
            )
        elif cfg.INPUT_MODE == "direct":
            self.preproc = None
        else:
            raise ValueError(f"Unknown INPUT_MODE '{cfg.INPUT_MODE}'. "
                              f"Choices: direct, preprocessed.")

        dims = [cfg.N_QUBITS] + list(cfg.CLASSICAL_HEAD_LAYERS)
        head_layers = []
        for i in range(len(dims) - 1):
            head_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                head_layers.append(nn.Tanh())
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x, t: shape (N, 1) each. Returns u: shape (N, 1)."""
        xt = torch.cat([x, t], dim=1)  # (N, 2), x in [-1,1], t in [0,1]

        if self.preproc is not None:
            xt = self.preproc(xt)

        # Rescale t from [0,1] to [-1,1] so both inputs use a symmetric
        # encoding range (angle-encoding gates are most well-conditioned
        # when their arguments are centered near zero).
        x_s = xt[:, 0:1]
        t_s = xt[:, 1:2] * 2.0 - 1.0
        xt_scaled = torch.cat([x_s, t_s], dim=1)

        q_out = self.qlayer(xt_scaled)     # (N, n_qubits)
        u = self.head(q_out)               # (N, 1)
        return u

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg) -> QAPINN:
    model = QAPINN(cfg)
    return model.to(device=cfg.DEVICE, dtype=cfg.DTYPE)
