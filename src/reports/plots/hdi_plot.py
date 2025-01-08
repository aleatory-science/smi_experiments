# Dataset
from src.sine_wave_data import VIS, IN, DOM_C1, DOM_C2
from src.sine_wave_data import sine_wave_fn

from matplotlib import pyplot as plt
from src.logger.reg_logger import ExpLogger
import numpy as np
from numpyro.diagnostics import hpdi
from src.reports.plots.util import IMG_DIR
import json

IMG_1D_DIR = IMG_DIR / "1d_bnn_hdi"
IMG_1D_DIR.mkdir(exist_ok=True, parents=True)


def ci_plot(eval_data, tr_data, pred_y, title="", add_legend=False, loc=""):
    ex, _ = eval_data
    ex = ex.squeeze()
    tx, ty = tr_data
    fig, ax = plt.subplots()
    pred_y = pred_y.squeeze()

    ax.spines[["right", "top"]].set_visible(False)
    ax.fill_between(
        np.linspace(*DOM_C1), -20, 20, alpha=0.05, facecolor="black", zorder=1
    )
    ax.axvline(DOM_C1[0], color="black", linewidth=3, zorder=1)
    ax.axvline(DOM_C1[1], color="black", linewidth=3, zorder=1)

    ax.fill_between(
        np.linspace(*DOM_C2), -20, 20, alpha=0.05, facecolor="black", zorder=1
    )
    ax.axvline(DOM_C2[0], color="black", linewidth=3, zorder=1)
    ax.axvline(DOM_C2[1], color="black", linewidth=3, zorder=1)

    ax.plot(ex, sine_wave_fn(np.array(ex)), "k--", linewidth=2)
    ax.plot(ex, pred_y.mean(0), color="black", label="Average", linewidth=4)

    ax.fill_between(
        ex, *hpdi(pred_y), alpha=0.4, color="black", interpolate=True, label="HDI"
    )

    ax.set_xlim([-2, 2])
    if "high" in loc.name:
        ax.set_ylim([-10, 15])
    if "low" in loc.name:
        ax.set_ylim([-7, 9])

    if add_legend:
        ax.legend(fontsize=18)
    if title:
        ax.set_title(title)

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.set_xticks([-2, -1, 0, 1, 2])
    # plt.tight_layout(pad=0.2)
    fig.tight_layout(pad=0.3)

    if loc:
        fig.savefig(loc)

    plt.close(fig)


def combinations():
    """Combinations of methods, model type and kernel to be tested."""
    for model_type in ["high", "low"]:
        for method in sorted(["nuts", "asvgd", "smi", "svgd", "ovi"]):
            match method:
                case "ovi" | "nuts":
                    yield model_type, method, "none"
                case "svgd" | "asvgd" | "smi":
                    yield model_type, method, "rbf"
                case "smi":
                    for kernel in ["ppk"]:
                        yield model_type, method, kernel


def sine_wave_hdi(logger, plot_format):
    """Plot highest predictive posterior density interval. Figure 3. in the article."""
    plt.rcParams.update({"font.size": 16})
    logs = logger.get_logs()

    img_dir = IMG_DIR / "1d_bnn_hdi"
    img_dir.mkdir(exist_ok=True, parents=True)

    for mt, method, kernel in combinations():
        y = np.load(logger.root / logs[mt][method][kernel]["vis"]["y_loc"][4])
        ci_plot(
            VIS.eval,
            IN.tr,
            y,
            loc=img_dir / f"{mt}_{method}.{plot_format}",
            add_legend=mt == "low" and method == "svgd",
        )


def draw_ci(ax, mt, eval_data, tr_data, pred_y):
    ex, _ = eval_data
    ex = ex.squeeze()
    tx, ty = tr_data
    pred_y = pred_y.squeeze()

    ax.spines[["right", "top"]].set_visible(False)
    ax.fill_between(
        np.linspace(*DOM_C1), -20, 20, alpha=0.05, facecolor="black", zorder=1
    )
    ax.axvline(DOM_C1[0], color="black", linewidth=1, zorder=1)
    ax.axvline(DOM_C1[1], color="black", linewidth=1, zorder=1)

    ax.fill_between(
        np.linspace(*DOM_C2), -20, 20, alpha=0.05, facecolor="black", zorder=1
    )
    ax.axvline(DOM_C2[0], color="black", linewidth=1, zorder=1)
    ax.axvline(DOM_C2[1], color="black", linewidth=1, zorder=1)

    ax.plot(ex, sine_wave_fn(np.array(ex)), "k--", linewidth=1)
    ax.plot(ex, pred_y.mean(0), color="black", label="Average", linewidth=1)

    ax.fill_between(
        ex, *hpdi(pred_y), alpha=0.4, color="black", interpolate=True, label="HDI"
    )

    ax.set_xlim([-2, 2])

    match mt:
        case "high":
            ax.set_ylim([-15, 15])
        case "low":
            ax.set_ylim([-9, 9])


def total(it):
    return sum(1 for _ in it)


def all_hdi(logger, plot_format):
    logs = logger.get_logs()
    root = logger.root

    with open(root / logs["exp_config"]) as f:
        r = json.load(f)["repeats"]

    # Draw seperate figure for high and low
    n_meth = total(combinations()) // 2
    # Each row is a repeat and each column is a method
    hfig, haxs = plt.subplots(nrows=r, ncols=n_meth, figsize=(1.7 * n_meth, 1.4 * r))
    lfig, laxs = plt.subplots(nrows=r, ncols=n_meth, figsize=(1.7 * n_meth, 1.4 * r))

    for i, (mt, method, kernel) in enumerate(combinations()):
        match mt:
            case "high":
                axs = haxs[:, i]
            case "low":
                axs = laxs[:, i % n_meth]  # We first enumerate all of high then low

        for j, loc in enumerate(logs[mt][method][kernel]["vis"]["y_loc"]):
            y = np.load(logger.root / loc)
            draw_ci(axs[j], mt, VIS.eval, IN.tr, y)

    r_last = r - 1  # correct for zero index
    methods = sorted(["nuts", "asvgd", "smi", "svgd", "ovi"])
    for i in range(r):
        for j in range(n_meth):
            ha, la = haxs[i, j], laxs[i, j]
            if i == 0:
                # Add method names to the first row
                ha.set_title(methods[j].upper())
                la.set_title(methods[j].upper())

            if i == r_last:
                # Add x label to last row
                ha.set_xlabel("x")
                la.set_xlabel("x")
            if j == 0:
                # Add y label to first column
                ha.set_ylabel("y")
                la.set_ylabel("y")

            # Remove tick labels from inner plots
            labelx = i == r_last  # Have x tick labels only in the last row
            labely = j == 0  # Have y tick labels only in first column

            ha.tick_params(labelleft=labely, labelbottom=labelx)
            la.tick_params(labelleft=labely, labelbottom=labelx)

    hfig.tight_layout()
    hfig.savefig(IMG_1D_DIR / "high_all.png")

    lfig.tight_layout()
    lfig.savefig(IMG_1D_DIR / "low_all.png")

    # Clean
    plt.close(hfig)
    plt.close(lfig)


if __name__ == "__main__":
    lppd_logger = ExpLogger("lppd_1d")
    lppd_logger.load_latest_logs()
    sine_wave_hdi(lppd_logger, "png")
    all_hdi(lppd_logger, "png")
