import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from PINN_WSGG.nn_wsgg import NN_WSGG, nn_wsgg_rte
from PINN_WSGG.plot_setup import setup


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "PINN_WSGG" / "data" / "wsgg.json"
PROFILES_PATH = DATA_DIR / "profiles.npy"
RESULTS_PATH = DATA_DIR / "results.npy"
FIGURE_PATH = BASE_DIR / "test_case.png"

DEVICE = torch.device("cpu")
DTYPE = torch.float32

PROFILE_COLUMNS = {
    "x": 0,
    "x_h2o": 1,
    "x_co2": 2,
    "x_co": 3,
    "temperature": 4,
    "pressure": 5,
}

def load_model(model_path: Path, device: torch.device, dtype: torch.dtype) -> NN_WSGG:
    payload = json.loads(model_path.read_text())

    hidden_dims = payload["hidden_dims"]
    x_mean = torch.tensor(payload["x_mean"], dtype=dtype)
    x_std = torch.tensor(payload["x_std"], dtype=dtype)

    model = NN_WSGG(hidden_dims, x_mean, x_std).to(device=device, dtype=dtype)

    state_dict = {}
    for key, value in payload.items():
        if key == "hidden_dims":
            continue
        state_dict[key] = torch.tensor(value, dtype=dtype)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_case_data(
    profiles_path: Path,
    results_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    profiles_np = np.load(profiles_path)
    results_np = np.load(results_path)

    if profiles_np.ndim != 2 or profiles_np.shape[1] != 6:
        raise ValueError(
            f"Expected profiles to have shape [N, 6], got {profiles_np.shape}"
        )
    if results_np.ndim != 2 or results_np.shape[1] != 3:
        raise ValueError(
            f"Expected results to have shape [N, 3], got {results_np.shape}"
        )

    profiles = torch.tensor(profiles_np, device=device, dtype=dtype)
    return profiles, results_np


def compute_relative_mae(reference: np.ndarray, prediction: np.ndarray) -> float:
    span = np.max(reference) - np.min(reference)
    if np.isclose(span, 0.0):
        return float(np.mean(np.abs(reference - prediction)))
    return float(np.mean(np.abs(reference - prediction)) / span)


def main() -> None:
    setup()

    model = load_model(MODEL_PATH, DEVICE, DTYPE)
    profiles, results = load_case_data(PROFILES_PATH, RESULTS_PATH, DEVICE, DTYPE)

    xs = profiles[:, PROFILE_COLUMNS["x"]]
    x_h2o_profiles = profiles[:, PROFILE_COLUMNS["x_h2o"]]
    x_co2_profiles = profiles[:, PROFILE_COLUMNS["x_co2"]]
    x_co_profiles = profiles[:, PROFILE_COLUMNS["x_co"]]
    temperature_profiles = profiles[:, PROFILE_COLUMNS["temperature"]]
    pressure_profiles = profiles[:, PROFILE_COLUMNS["pressure"]]

    T_l = torch.tensor(300.0, device=DEVICE, dtype=DTYPE)
    T_r = torch.tensor(300.0, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        pred_G, pred_q, pred_dq = nn_wsgg_rte(
            model,
            xs,
            temperature_profiles,
            x_h2o_profiles,
            x_co2_profiles,
            x_co_profiles,
            pressure_profiles,
            T_l,
            T_r,
        )

    plot_xs = xs.detach().cpu().numpy()
    dq_reference = results[:, 2] / 1e6
    dq_prediction = pred_dq.detach().cpu().numpy() / 1e6
    dq_rel_mae = compute_relative_mae(results[:, 2], pred_dq.detach().cpu().numpy())

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.plot(plot_xs, dq_reference, "b", linewidth=1.2, label="LBL")
    ax.plot(
        plot_xs,
        dq_prediction,
        "r--",
        linewidth=1.2,
        label="PINN-WSGG",
    )

    min_val = np.min(dq_reference)
    max_val = np.max(dq_reference)
    diff_val = max_val - min_val
    if np.isclose(diff_val, 0.0):
        diff_val = 1e-12
    ax.set_ylim([min_val - diff_val * 0.1, max_val + diff_val * 1.0])
    ax.set_xlim(np.min(plot_xs), np.max(plot_xs))
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel(r"$\nabla\cdot q$ [MW/m$^3$]")
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300)

    print(f"dq relative MAE: {dq_rel_mae:.6f}")
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
