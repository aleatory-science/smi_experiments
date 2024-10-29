import seaborn as snb
from pyro.distributions import MultivariateNormal, Bernoulli
from pyro import sample, plate, set_rng_seed
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, ConnectionPatch, ConnectionStyle
import matplotlib
from src.report.plots.util import IMG_DIR

matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}\usepackage{nicefrac}\usepackage{fixcmex}"
)

root = IMG_DIR / "methods"
root.mkdir(exist_ok=True, parents=True)

plt.clf()
plt.rcParams["text.usetex"] = True
snb.set_theme(style="white")
set_rng_seed(1)
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_facecolor("lavender")
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(2)


def model(n):
    with plate("data", n):
        c1 = sample(
            "c1",
            MultivariateNormal(
                torch.tensor([-0.5, 3.0]), torch.tensor([[2.5, 1.0], [1.0, 1.0]])
            ),
        )
        c2 = sample(
            "c2",
            MultivariateNormal(
                torch.tensor([0.5, -0.5]), torch.tensor([[0.7, 0.2], [0.2, 0.3]])
            ),
        )
        idx = sample("idx", Bernoulli(torch.tensor(0.7)))
        return torch.where(idx.type(torch.bool).reshape(-1, 1), c1, c2)


x, y = model(50).T
data = dict(x=x, y=y)
g1 = snb.kdeplot(
    data=data, x="x", y="y", fill=True, levels=8, cmap="Blues", zorder=1, ax=ax
)
g2 = snb.kdeplot(
    data=data, x="x", y="y", fill=False, levels=8, color="black", zorder=5, ax=ax
)
g2.set(xticklabels=[])
g2.set(xlabel=None)
g2.set(yticklabels=[])
g2.set(ylabel=None)
g2.tick_params(bottom=False)  # remove the ticks
g2.tick_params(left=False)  # remove the ticks
plt.tight_layout(pad=0.0)
ax.set_xlim([-6, 5.5])
ax.set_ylim([-3.5, 7])

ax.scatter(
    *torch.stack([x[:10], y[:10]]),
    s=170,
    color="#fa6b6b",
    linewidth=3,
    edgecolor="#eb3131",
    zorder=10,
)
ax.text(4, 4.6, r"$\boldsymbol{\theta}$", size=60, zorder=20)

ax.text(
    -6.25,
    -5,
    r"$p(\boldsymbol{\theta}|\mathcal{D}) \approx \nicefrac{1}{m} \sum_{\ell=1}^m \delta_{\boldsymbol{\theta}_\ell}(\boldsymbol{\theta})$",
    size=32,
    zorder=20,
)
plt.gca().set_aspect("equal", adjustable="box")

fig.subplots_adjust(top=0.8)
# fig.suptitle('Stein particle approximation', size=50)
plt.savefig(root / "svgd.png", bbox_inches="tight", pad_inches=0.1)

plt.clf()
plt.rcParams["text.usetex"] = True
x_ = np.linspace(*g2.get_xlim(), 1000)
y_ = np.linspace(*g2.get_ylim(), 1000)
x_, y_ = np.meshgrid(x_, y_)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[1]
ax.set_facecolor("lavender")
for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(3)

ax.set_xlim([-6, 5.5])
ax.set_ylim([-3.5, 7])
ax.text(4, 4.6, r"$\boldsymbol{\theta}$", size=60, zorder=20)

g1 = snb.kdeplot(data=data, x="x", y="y", fill=True, levels=8, cmap="Blues", zorder=1)
g2 = snb.kdeplot(data=data, x="x", y="y", fill=False, levels=8, color="black", zorder=5)
g2.set(xticklabels=[])
g2.set(xlabel=None)
g2.set(yticklabels=[])
g2.set(ylabel=None)
g2.tick_params(bottom=False)  # remove the ticks
g2.tick_params(left=False)  # remove the ticks
plt.tight_layout(pad=0.0)

ell = Ellipse(
    xy=[0.5, -0.5],
    width=3,
    height=2.5,
    angle=70,
    facecolor="#fa6b6b",
    edgecolor="#eb3131",
    lw=3,
)
ax.add_artist(ell)
ell = Ellipse(
    xy=[-1, 2.5],
    width=5,
    height=2.5,
    angle=35,
    facecolor="#fa6b6b",
    edgecolor="#eb3131",
    lw=3,
)
ax.add_artist(ell)


ax = axs[0]
plt.tight_layout(pad=0.0)
ell = Ellipse(
    xy=[0, 3],
    width=8,
    height=5,  # angle=35,
    color="lightgrey",
)
#  color='#6ab873')
ax.add_artist(ell)
p1 = torch.tensor([x[15], y[15]])
p2 = torch.tensor([x[16], y[16]])
ax.scatter(
    *torch.stack([p1, p2]),
    s=170,
    color="#fa6b6b",
    linewidth=3,
    edgecolor="#eb3131",
    zorder=10,
)
ax.set_xlim([-4.5, 5.5])
ax.set_ylim([-3.5, 6])
ax.set_axis_off()

arrow = ConnectionPatch(
    p2 + torch.tensor([0.1, 0.0]),
    [-1.85, 3.5],
    coordsA=axs[0].transData,
    coordsB=axs[1].transData,
    # Default shrink parameter is 0 so can be omitted
    color="black",
    arrowstyle="-",  # "normal" arrow
    mutation_scale=30,  # controls arrow head size
    linewidth=3,
    linestyle="dashed",
    connectionstyle=ConnectionStyle.Arc3(rad=-0.3),
)
fig.patches.append(arrow)

arrow = ConnectionPatch(
    p1 + torch.tensor([0.15, -0.05]),
    (-0.8, -0.7),
    coordsA=axs[0].transData,
    coordsB=axs[1].transData,
    # Default shrink parameter is 0 so can be omitted
    color="black",
    arrowstyle="-",  # "normal" arrow
    mutation_scale=30,  # controls arrow head size
    linewidth=3,
    linestyle="dashed",
    connectionstyle=ConnectionStyle.Arc3(rad=0.4),
)
fig.patches.append(arrow)
plt.annotate(
    r"$q(\boldsymbol{\theta}|\boldsymbol{\psi}_1)$",
    [375, 500],
    size=28,
    xycoords="figure pixels",
    va="center",
)
plt.annotate(
    r"$q(\boldsymbol{\theta}|\boldsymbol{\psi}_2)$",
    [375, 200],
    size=28,
    xycoords="figure pixels",
    va="center",
)

ax.text(-2, 3.5, r"$\boldsymbol{\psi}$", size=60, zorder=20, color="black")

ax.text(
    0,
    -4.4,
    r"$p(\boldsymbol{\theta}|\mathcal{D}) \approx \nicefrac{1}{m} \sum_{\ell=1}^m q(\boldsymbol{\theta}|\boldsymbol{\psi}_\ell)$",
    size=36,
    zorder=20,
)

plt.gca().set_aspect("equal", adjustable="box")

# fig.subplots_adjust(top=0.8)
# fig.suptitle('Stein mixture approximation', size=50)


plt.savefig(root / "smi.png", bbox_inches="tight", pad_inches=0.1)
