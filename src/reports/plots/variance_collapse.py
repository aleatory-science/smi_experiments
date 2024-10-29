import numpy as np
import dill
import json
from matplotlib import pyplot as plt
from src.reports.plots.util import IMG_DIR

from collections import defaultdict

METHOD_LINESTYLES = {
    "svgd": {20: ("dashdot", "black")},
    "asvgd": {20: ("dashed", "grey")},
    "smi": {20: ("dotted", "darkgrey"), 1: ("dotted", "lightgrey")},
}
RT_MARKERS = {0.001: "v", 0.1: ">", 1.0: "o", 10.0: "<", 100.0: "^"}


def _get_measures(name, root, logs, methods):
    rts = {}
    ns = {}
    inits = {}
    meass = defaultdict(list)
    for method in methods:
        rts[method] = defaultdict(list)
        inits[method] = defaultdict(list)
        ns[method] = defaultdict(set)

        for i, loc in enumerate(logs["bnn"][method]["rbf"]["none"]["artifact"]):
            hparams = dill.load((root / loc).open("br"))["hyper_params"]
            rt = hparams["repulsion_temperature"]
            n = hparams["num_particles"]
            rad = hparams["init_radius"]

            rts[method][rt].append(i)
            inits[method][rad].append(i)
            ns[method][n].add(i)

        for loc in logs["bnn"][method]["rbf"]["none"][name]:
            meass[method].append(np.load(root / loc))
    return meass, rts, ns, inits


def variance_plot(logger, add_std):
    plt.rcParams.update({"font.size": 14})
    root = logger.root
    logs = logger.get_logs()

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    log_vax_p20, vax_p20, vax_p1 = axs

    methods = list(logs["bnn"].keys())

    # Structure data
    mdims = json.load((root / logs["exp_config"]).open())["model_dimensions"]

    # Draw ideal performance
    log_vax_p20.hlines(1, min(mdims), max(mdims), color="black", lw=2)
    vax_p20.hlines(1, min(mdims), max(mdims), color="black", lw=2)
    vax_p1.hlines(1, min(mdims), max(mdims), color="black", lw=2)

    vs, rts, ns, inits = _get_measures("var", root, logs, methods)

    # Build 20 particle plots
    for i, method in enumerate(sorted(methods)):
        (linestyle, color) = METHOD_LINESTYLES[method][20]
        for rt, marker in RT_MARKERS.items():
            idxs = rts[method][rt]

            if method == "smi":
                # Pick 20 particles results
                idxs = [idx for idx in idxs if idx in inits[method][2]]
                idxs = [idx for idx in idxs if idx in ns[method][20]]
            else:
                idxs = [idx for idx in idxs if idx in inits[method][20]]

            dim_mean_var = np.array([np.mean(vs[method][idx]) for idx in idxs])
            dim_std_var = np.array([np.std(vs[method][idx]) for idx in idxs])

            log_vax_p20.plot(
                mdims,
                dim_mean_var,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=5 - 2 * i,
                markersize=7 - 2 * i,
            )

            vax_p20.plot(
                mdims,
                dim_mean_var,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=5 - 2 * i,
                markersize=7 - 2 * i,
            )

            if add_std:
                vax_p20.fill_between(
                    mdims,
                    dim_mean_var - dim_std_var,
                    dim_mean_var + dim_std_var,
                    color=color,
                    alpha=0.2,
                )

    # Build 1 particle SMI (all RT) plot vs ASVGD, SVGD 20 particle RT=1
    for i, method in enumerate(sorted(methods)):
        if method == "smi":
            (linestyle, color) = METHOD_LINESTYLES[method][1]
        else:
            (linestyle, color) = METHOD_LINESTYLES[method][20]

        for rt, marker in RT_MARKERS.items():
            # Only include RT=1 for ASVGD and SVGD
            if method != "smi" and rt != 1.0:
                continue

            idxs = rts[method][rt]
            if method == "smi":
                # Pick single particle results
                idxs = [idx for idx in idxs if idx in inits[method][2]]
                idxs = [idx for idx in idxs if idx in ns[method][1]]
            else:
                idxs = [idx for idx in idxs if idx in inits[method][20]]

            dim_mean_var = np.array([np.mean(vs[method][idx]) for idx in idxs])
            dim_std_var = np.array([np.std(vs[method][idx]) for idx in idxs])

            idxs = rts[method][rt]
            vax_p1.plot(
                mdims,
                dim_mean_var,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=5 - 2 * i,
                markersize=7 - i - i * (i > 1),
            )

            if add_std:
                vax_p1.fill_between(
                    mdims,
                    dim_mean_var - dim_std_var,
                    dim_mean_var + dim_std_var,
                    color=color,
                    alpha=0.2,
                )

    log_vax_p20.set_yscale("log")
    vax_p20.set_ylim([0.6, 1.4])
    vax_p20.set_xscale("log")

    for ax in (log_vax_p20, vax_p1):  # axs.flatten():
        ax.set_xticks([0, 10, 20, 40, 60, 80, 100])
        ax.set_xticks(mdims, minor=True)

    for ax in axs.flatten():
        ax.set_xlabel("Dimensionality")
        ax.set_xlim([min(mdims), max(mdims)])
        ax.set_xticks(mdims, minor=True)

    log_vax_p20.set_ylabel("Avg. dimension marginal variance")

    # Legend for repulsive temperature

    # Creating custom proxy artists for markers
    markers = [
        plt.Line2D(
            [0],
            [0],
            marker=RT_MARKERS[rt],
            color="w",
            markerfacecolor="k",
            markersize=8,
            linestyle="None",
        )
        for rt in sorted(RT_MARKERS, reverse=True)
    ]
    # Legend for marker styles
    marker_legend = vax_p1.legend(
        markers,
        [rf"$\alpha$={rt}" for rt in sorted(RT_MARKERS, reverse=True)],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.95),
    )

    # Add the marker legend to the plot
    vax_p1.add_artist(marker_legend)

    # Legend for linestyles
    # Creating custom proxy artists for linestyles
    linestyles = [plt.Line2D([0], [0], color="black", lw=2, linestyle="solid")]
    line_cap = ["Actual"]

    old_params = {"mathtext.default": plt.rcParams["mathtext.default"]}
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)

    for i, method in enumerate(sorted(methods)):
        line = plt.Line2D(
            [0],
            [0],
            color=METHOD_LINESTYLES[method][20][1],
            lw=5 - 2 * i,
            linestyle=METHOD_LINESTYLES[method][20][0],
        )
        linestyles.append(line)
        line_cap.append(f"${method.upper()}" + "_{20}$")
        if method == "smi":
            line = plt.Line2D(
                [0],
                [0],
                color=METHOD_LINESTYLES[method][1][1],
                lw=5 - 2 * i,
                linestyle=METHOD_LINESTYLES[method][1][0],
            )
            linestyles.append(line)
            line_cap.append(f"${method.upper()}_1$")

    linestyle_legend = vax_p1.legend(
        linestyles, line_cap, loc="lower right", bbox_to_anchor=(1.0, 0.1)
    )

    vax_p1.add_artist(linestyle_legend)

    save_dir = IMG_DIR / "var"
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir / "variance_collapse.png")
    plt.rcParams.update(old_params)


if __name__ == "__main__":
    from src.logger import ExpLogger

    logger = ExpLogger("var")
    logger.load_latest_logs()

    variance_plot(logger, add_std=False)
