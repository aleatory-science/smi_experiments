from matplotlib import pyplot as plt
from src.sine_wave_data import IN, BETWEEN, ENTIRE, VIS, sine_wave_fn, DOM_C1, DOM_C2
from src.reports.plots.util import IMG_DIR
import numpy as np

plt.rcParams.update({"font.size": 20})


def fn(name, x, y, color):
    fig, ax = plt.subplots()  # figsize=(7,5))
    ax.spines[["right", "top"]].set_visible(False)

    ax.scatter(
        x, y, marker="x", s=150, color="black", linewidth=5, zorder=3, label=name
    )
    ax.fill_between(
        np.linspace(*DOM_C1), -10, 10, alpha=0.05, facecolor="black", zorder=1
    )
    ax.axvline(DOM_C1[0], color="black", linewidth=3, zorder=1)
    ax.axvline(DOM_C1[1], color="black", linewidth=3, zorder=1)

    ax.fill_between(
        np.linspace(*DOM_C2), -10, 10, alpha=0.05, facecolor="black", zorder=1
    )
    ax.axvline(DOM_C2[0], color="black", linewidth=3, zorder=1)
    ax.axvline(DOM_C2[1], color="black", linewidth=3, zorder=1)

    ax.plot(VIS.eval[0], sine_wave_fn(VIS.eval[0]), "k--", linewidth=4, zorder=2)

    ax.set_ylim([-6, 8.2])
    ax.set_xlim([-2, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return fig


def sine_wave_data(plot_format):
    names = ["train", "in", "between", "entire"]
    data = [IN.tr, IN.eval, BETWEEN.eval, ENTIRE.eval]
    colors = ["blue", "purple", "green", "orange"]

    img_dir = IMG_DIR / "sinewave"
    img_dir.mkdir(exist_ok=True, parents=True)

    for n, (x, y), c in zip(names, data, colors):
        fig = fn(n.capitalize(), x, y, c)
        plt.tight_layout(pad=0.2)
        fig.savefig(img_dir / f"{n}.{plot_format}")
        plt.close(fig)


# sine_wave_data(IMG_DIR, "png")
