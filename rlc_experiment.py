#!/usr/bin/env python3
"""Reproduce the periodic cubic-capacitor experiment in Section 5.

Run from any directory: python code/rlc_experiment.py
The script writes vector PDF figures and complete numeric data next to this file.
Circuit parameters and the sampling grid match the supplied physical-unit code.
Time, voltage, current, flux, and charge are stored in SI units.
"""

from pathlib import Path
import json
import platform
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

PARAMETERS = dict(N=100, T=0.02, L=0.005, R=5.0, C0=0.004, C3=0.004,
                  source_amplitude=1.0, alpha=1.0, beta=1000.0, theta=0.0,
                  relative_tolerance=1e-12, maximum_iterations=2000)
V_REF, I_REF = 1.0, 1.0  # Fixed 1 V and 1 A scales for equation residuals.


def antisymmetric(z):
    """Restore an invariant linear subspace against floating-point roundoff."""
    return 0.5 * (z - np.roll(z, z.shape[-1] // 2, axis=-1))


def cubic_resolvent(rhs, beta, c0, c3):
    """The unique real solution of (1+beta*c0)*v+beta*c3*v**3=rhs."""
    if c3 == 0:
        return rhs / (1.0 + beta * c0)
    p = (1.0 + beta * c0) / (beta * c3)
    q = rhs / (beta * c3)
    return 2.0 * np.sqrt(p / 3.0) * np.sinh(
        np.arcsinh(1.5 * q / p * np.sqrt(3.0 / p)) / 3.0)


def main():
    p = PARAMETERS
    n, period = p["N"], p["T"]
    assert n % 4 == 0, "N must be divisible by four to exclude both null modes."
    h = period / n
    time = h * np.arange(n)
    symbol = 1j * np.sin(2.0 * np.pi * np.arange(n) / n) / h
    a, b, theta = p["alpha"], p["beta"], p["theta"]
    inductance, resistance, c0, c3 = p["L"], p["R"], p["C0"], p["C3"]
    source = antisymmetric(p["source_amplitude"] * np.sin(2.0 * np.pi * time / period))

    def derivative(z):
        return ((np.roll(z, -1, axis=-1) - np.roll(z, 1, axis=-1)) / (2.0 * h))

    def shifted_derivative(rhs):
        return np.fft.ifft(np.fft.fft(rhs) / (1.0 + a * symbol)).real

    def iterate(z):
        # z=(phi, v_R, q, i_L, i_R, v), with one row per proof block.
        phi, vr, charge, il, ir, voltage = z
        yp = np.stack((shifted_derivative(phi - a * voltage + a * source),
                       (vr + a * voltage) / (1.0 + a),
                       shifted_derivative(charge + a * (il - ir))))
        ybar = (1.0 + theta) * yp - theta * z[:3]
        xp = np.stack(((il + b * ybar[0]) / (1.0 + b * inductance),
                       (ir + b * ybar[1]) / (1.0 + b * resistance),
                       cubic_resolvent(voltage + b * ybar[2], b, c0, c3)))
        return antisymmetric(np.concatenate((yp, xp)))

    minimum_derivative_gain = np.sin(2.0 * np.pi / n) / h
    ell_a = np.array([1.0 / np.sqrt(1.0 + (a * minimum_derivative_gain) ** 2),
                      1.0 / (1.0 + a),
                      1.0 / np.sqrt(1.0 + (a * minimum_derivative_gain) ** 2)])
    ell_b = np.array([1.0 / (1.0 + b * inductance),
                      1.0 / (1.0 + b * resistance),
                      1.0 / (1.0 + b * c0)])
    la, lb = np.diag(ell_a), np.diag(ell_b)
    kh = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    ctheta = (1.0 + theta) * la + abs(theta) * np.eye(3)
    gain = np.block([[la, a * la @ kh],
                     [b * lb @ ctheta, lb + (1.0 + theta) * a * b * lb @ la @ kh]])
    eigenvalues, eigenvectors = np.linalg.eig(gain)
    perron_index = int(np.argmax(eigenvalues.real))
    rho = float(eigenvalues[perron_index].real)
    weights = eigenvectors[:, perron_index].real
    weights *= np.sign(weights[np.argmax(abs(weights))])
    weights /= np.max(weights)
    assert np.all(weights > 0.0) and rho < 1.0
    assert np.max(abs(gain @ weights - rho * weights)) < 1e-12

    def block_lengths(z):
        return np.sqrt(h * np.sum(z * z, axis=1))

    def block_norm(z):
        return float(np.max(block_lengths(z) / weights))

    def product_norm(z):
        return float(np.sqrt(h * np.sum(z * z)))

    z = np.zeros((6, n))
    initial_step = iterate(z) - z
    initial_w = block_norm(initial_step)
    initial_2 = product_norm(initial_step)
    rows = []
    history = [z.copy()]
    for k in range(p["maximum_iterations"] + 1):
        zp = iterate(z)
        step = zp - z
        rw = block_norm(step) / initial_w
        r2 = product_norm(step) / initial_2
        rows.append([k, rw, r2, rho ** k])
        if rw <= p["relative_tolerance"]:
            break
        z = zp
        history.append(z.copy())
    else:
        raise RuntimeError("The certified run did not reach the selected tolerance.")
    history = np.asarray(history)
    residuals = np.asarray(rows)
    terminal = z.copy()
    # Finish well beyond the plotting tolerance to obtain a reference solution.
    for _ in range(250):
        reference = iterate(z)
        if product_norm(reference - z) < 1e-15:
            z = reference
            break
        z = reference
    reference = z

    def original_residual(z):
        il, ir, voltage = z[3:]
        return np.stack(((inductance * derivative(il) + voltage - source) / V_REF,
                         (resistance * ir - voltage) / V_REF,
                         (derivative(c0 * voltage + c3 * voltage ** 3) - il + ir) / I_REF))

    def lifted_residual(z):
        phi, vr, charge, il, ir, voltage = z
        return np.stack((derivative(phi) + voltage - source,
                         vr - voltage,
                         derivative(charge) - il + ir,
                         inductance * il - phi,
                         resistance * ir - vr,
                         c0 * voltage + c3 * voltage ** 3 - charge))

    # Fits exclude the initial transient and the roundoff plateau.
    fit_mask = (residuals[:, 1] < 1e-4) & (residuals[:, 1] > 1e-9)
    slope = np.polyfit(residuals[fit_mask, 0], np.log(residuals[fit_mask, 1]), 1)[0]
    observed_factor = float(np.exp(slope))
    gamma_a = a * ell_a / (1.0 - ell_a)
    gamma_b = b * ell_b / (1.0 - ell_b)
    chi = 1.0 + theta + abs(theta)
    reduced_gain = chi * np.diag(gamma_a * gamma_b) @ kh
    limit_gain = chi * np.diag([1.0 / (minimum_derivative_gain * inductance),
                                1.0 / resistance,
                                1.0 / (minimum_derivative_gain * c0)]) @ kh
    # Physical capacitor current is computed from the voltage constitutive law.
    # D_N q^k is an auxiliary lifted quantity before convergence and is not used.
    capacitor_voltage = history[:, 5, :]
    capacitor_current = derivative(c0 * capacitor_voltage + c3 * capacitor_voltage ** 3)
    reference_capacitor_current = derivative(c0 * reference[5] + c3 * reference[5] ** 3)
    snapshots = sorted(set(range(0, min(60, k) + 1, 2))
                       | set(range(64, k + 1, 4)) | {k})
    certificate = {
        "parameters": p,
        "parameter_units": {"T": "s", "L": "H", "R": "ohm", "C0": "F",
                            "C3": "C/V^3", "source_amplitude": "V"},
        "source_frequency_hz": float(1.0 / period),
        "sampling_step_seconds": float(h),
        "state_units": ["Wb", "V", "C", "A", "A", "V"],
        "equation_reference_scales": {"voltage_V": V_REF, "current_A": I_REF},
        "minimum_derivative_gain": float(minimum_derivative_gain),
        "ell_A": ell_a.tolist(), "ell_B": ell_b.tolist(),
        "gain_matrix": gain.tolist(), "perron_weights": weights.tolist(),
        "certified_factor": rho,
        "reduced_spectral_radius": float(max(abs(np.linalg.eigvals(reduced_gain)))),
        "large_stepsize_spectral_radius": float(max(abs(np.linalg.eigvals(limit_gain)))),
        "first_iteration_below_tolerance": int(k),
        "relative_fixed_point_residual_weighted": float(residuals[-1, 1]),
        "relative_fixed_point_residual_product": float(residuals[-1, 2]),
        "observed_tail_factor_weighted": observed_factor,
        "fit_iteration_range": [int(residuals[fit_mask, 0][0]), int(residuals[fit_mask, 0][-1])],
        "relative_original_equation_residual": product_norm(original_residual(terminal)) / product_norm(source[None, :]),
        "relative_lifted_equation_residual": product_norm(lifted_residual(terminal)) / product_norm(source[None, :]),
        "absolute_original_equation_residual": product_norm(original_residual(terminal)),
        "maximum_pointwise_original_equation_residual": float(np.max(abs(original_residual(terminal)))),
        "maximum_pointwise_kvl_residual_V": float(np.max(abs(original_residual(terminal)[0])) * V_REF),
        "maximum_pointwise_resistor_residual_V": float(np.max(abs(original_residual(terminal)[1])) * V_REF),
        "maximum_pointwise_kcl_residual_A": float(np.max(abs(original_residual(terminal)[2])) * I_REF),
        "absolute_lifted_equation_residual": product_norm(lifted_residual(terminal)),
        "terminal_fixed_point_error_weighted": block_norm(terminal - reference),
        "absolute_residual_error_bound": block_norm(iterate(terminal) - terminal) / (1.0 - rho),
        "peak_voltage": float(np.max(abs(reference[5]))),
        "peak_inductor_current": float(np.max(abs(reference[3]))),
        "peak_resistor_current": float(np.max(abs(reference[4]))),
        "peak_capacitor_current": float(np.max(abs(reference_capacitor_current))),
        "max_nonlinear_linear_charge_ratio": float(c3 * np.max(reference[5] ** 2) / c0),
        "subspace_defect": float(np.max(abs(reference + np.roll(reference, n // 2, axis=1)))),
        "maximum_relative_bound_violation": float(np.max(residuals[:, 1] - residuals[:, 3])),
        "plotted_iteration_indices": snapshots,
        "plotted_final_iteration": int(k),
        "plotted_current_definition": "i_C^k = D_N Q(v^k)",
        "plotted_time_coordinate": "time in milliseconds",
        "saved_iteration_indices": list(range(len(history))),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "matplotlib": matplotlib.__version__, "system": platform.system(),
                        "machine": platform.machine()},
    }
    with (DATA / "rlc_summary.json").open("w") as stream:
        json.dump(certificate, stream, indent=2)
    np.savetxt(DATA / "rlc_residuals.csv", residuals, delimiter=",",
               header="iteration,relative_weighted_residual,relative_product_residual,certified_bound", comments="")
    np.savetxt(DATA / "rlc_waveforms.csv",
               np.column_stack((time, source, reference.T, reference_capacitor_current)), delimiter=",",
               header="time,source,flux,resistor_voltage,charge,inductor_current,resistor_current,voltage,capacitor_current", comments="")
    np.savez_compressed(DATA / "rlc_iterates.npz", time=time, iterates=history,
                        iteration=np.arange(len(history)), snapshot_indices=np.asarray(snapshots),
                        capacitor_voltage=capacitor_voltage, capacitor_current=capacitor_current,
                        terminal=terminal, reference=reference, weights=weights, gain=gain)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.0,
                         "axes.labelsize": 10.0, "axes.titlesize": 10.0,
                         "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
                         "legend.fontsize": 9.0, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.bbox": "tight"})
    green, bluegreen, final_color = "#58BD45", "#008C9C", "#006775"
    history_cmap = LinearSegmentedColormap.from_list(
        "green_to_teal", [(0.0, green), (0.28, "#20B45A"),
                          (0.88, bluegreen), (1.0, final_color)])
    history_norm = Normalize(vmin=0, vmax=k)
    plot_time = 1000.0 * np.r_[time, period]
    final_handle = Line2D([0], [0], color=final_color, lw=2.0,
                          label=rf"Final iterate: $k={k}$")

    def closed(values):
        return np.r_[values, values[0]]

    def save_figure(fig, name):
        fig.savefig(FIGURES / f"{name}.pdf")
        fig.savefig(FIGURES / f"{name}.png", dpi=180)
        plt.close(fig)

    def draw_trajectories(axis, values, title, ylabel, compact=False):
        for idx in snapshots:
            if idx == k:
                continue
            axis.plot(plot_time, closed(values[idx]), color=history_cmap(history_norm(idx)),
                      alpha=0.82, linewidth=0.65 if compact else 0.85, zorder=2)
        # The label k is exact: plot z^k, not the subsequently refined reference.
        axis.plot(plot_time, closed(values[k]), color=final_color,
                  linewidth=1.9 if compact else 2.2, zorder=4)
        axis.set(xlim=(0.0, 1000.0 * period), ylabel=ylabel)
        axis.set_xticks(np.linspace(0.0, 1000.0 * period, 5))
        axis.set_title(title, loc="left", fontweight="bold", fontsize=9 if compact else 10)
        axis.grid(True, color="#DFE7E8", linewidth=0.6)

    def iteration_colorbar(fig, cax, compact=False):
        mapper = plt.cm.ScalarMappable(norm=history_norm, cmap=history_cmap)
        colorbar = fig.colorbar(mapper, cax=cax,
                               ticks=np.unique(np.rint(np.linspace(0, k, 5)).astype(int)))
        colorbar.set_label("Iteration $k$", fontsize=8 if compact else 10, labelpad=6)
        colorbar.ax.tick_params(labelsize=7.5 if compact else 9)
        colorbar.outline.set_linewidth(0.5)

    def draw_residuals(axis, compact=False):
        axis.semilogy(residuals[:, 0], residuals[:, 1], color=final_color, lw=1.9,
                      label="Perron block maximum")
        axis.semilogy(residuals[:, 0], residuals[:, 2], color=green, lw=1.6,
                      linestyle="-.", label="Product $L^2$")
        axis.semilogy(residuals[:, 0], residuals[:, 3], color="#7B8794", lw=1.3,
                      linestyle="--", label=r"Certified $\rho(G_0)^k$")
        axis.set(xlim=(0, k), ylim=(5e-13, 1.5), xlabel="Iteration $k$",
                 ylabel="Relative fixed-point residual")
        axis.grid(True, which="major", color="#DFE7E8", linewidth=0.6)
        if compact:
            axis.tick_params(labelsize=8)
            axis.xaxis.label.set_size(9)
            axis.yaxis.label.set_size(9)

    # Reference-style standalone: two vertically stacked physical waveforms.
    fig = plt.figure(figsize=(7.0, 5.4))
    ax_voltage = fig.add_axes([0.13, 0.55, 0.70, 0.34])
    ax_current = fig.add_axes([0.13, 0.12, 0.70, 0.34], sharex=ax_voltage)
    draw_trajectories(ax_voltage, capacitor_voltage, "(a) Capacitor voltage", r"$v^k$ [V]")
    draw_trajectories(ax_current, capacitor_current, "(b) Capacitor current", r"$i_C^k=D_NQ(v^k)$ [A]")
    ax_voltage.tick_params(labelbottom=False)
    ax_current.set_xlabel("Time [ms]")
    iteration_colorbar(fig, fig.add_axes([0.87, 0.12, 0.025, 0.77]))
    fig.legend(handles=[final_handle], loc="upper center", bbox_to_anchor=(0.49, 0.995),
               frameon=False)
    save_figure(fig, "rlc_iterates")

    # Standalone residuals retain exactly the same data and rigorous envelope.
    fig = plt.figure(figsize=(6.8, 3.2))
    ax = fig.add_axes([0.13, 0.16, 0.84, 0.65])
    draw_residuals(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.99),
               ncol=3, frameon=False, fontsize=8.7, columnspacing=1.6)
    save_figure(fig, "rlc_residuals")

    # Compact manuscript composite with the previous approximately 6.5x5 footprint.
    fig = plt.figure(figsize=(6.8, 5.2))
    ax_voltage = fig.add_axes([0.10, 0.58, 0.33, 0.29])
    ax_current = fig.add_axes([0.54, 0.58, 0.33, 0.29])
    ax_residual = fig.add_axes([0.10, 0.11, 0.77, 0.29])
    draw_trajectories(ax_voltage, capacitor_voltage, "(a) Capacitor voltage", r"$v^k$ [V]", compact=True)
    draw_trajectories(ax_current, capacitor_current, "(b) Capacitor current", r"$i_C^k$ [A]", compact=True)
    for axis in [ax_voltage, ax_current]:
        axis.set_xlabel("Time [ms]", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.yaxis.label.set_size(9)
    iteration_colorbar(fig, fig.add_axes([0.91, 0.58, 0.016, 0.29]), compact=True)
    fig.legend(handles=[final_handle], loc="upper center", bbox_to_anchor=(0.49, 0.995),
               frameon=False, fontsize=9)
    draw_residuals(ax_residual, compact=True)
    ax_residual.set_title("(c) Fixed-point residuals", loc="left", fontweight="bold", fontsize=9)
    handles, labels = ax_residual.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.49, 0.505),
               ncol=3, frameon=False, fontsize=7.7, columnspacing=1.4)
    save_figure(fig, "rlc_convergence")

    # Co-timed final signals on one axis with fixed physical reference scales.
    fig = plt.figure(figsize=(6.8, 2.9))
    ax = fig.add_axes([0.12, 0.19, 0.75, 0.62])
    vline, = ax.plot(plot_time, closed(capacitor_voltage[k]) / V_REF,
                    color=final_color, lw=2.0, label=r"$v/(1\,\mathrm{V})$")
    iline, = ax.plot(plot_time, closed(capacitor_current[k]) / I_REF,
                    color=bluegreen, lw=1.8, linestyle="--",
                    label=r"$i_C/(1\,\mathrm{A})$")
    ax.set(xlim=(0, 1000.0 * period), xlabel="Time [ms]",
           ylabel="Value / reference scale")
    ax.grid(True, color="#DFE7E8", linewidth=0.6)
    fig.legend(handles=[vline, iline], loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=2, frameon=False)
    save_figure(fig, "rlc_voltage_current")
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
