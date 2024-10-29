# Stein Mixture Inference 
![Stein mixture inference uses a mixture model (with uniform weights) of $m$ guides $q(\theta|\psi_\ell)$, parameterized by particles $\psi_\ell$ to approximate $p(\theta|\mathcal{D})$. As a result, Stein mixture inference approximates a Bayesian posterior with a richer model that alleviates variance collapse in higher dimensional posteriors.](./images/smi.png)

This repo includes the experimental code for ELBOing Stein: Variational Bayes with Stein Mixture Inference. Stein mixture inference is available as an inference engine in [NumPyro](https://num.pyro.ai/en/stable/).


## Installation
The experimental setup assumes a GPU device is available. To set up the project, ensure you have a Python `>3.8` and the latest version of `pip`. To check use:
``` shell
> python --version
Python 3.12.6
> pip --version
pip 24.2 ...
```

Setup a virtual env and install requirements: 
``` shell
> python -m venv .venv
> . .venv/bin/activate
> pip install -r requirements.txt
```

If [datasets](datasets) is missing
``` shell
> git submodule add git@github.com:svendoc/datasets.git
```

## Run experiments
To run our experiments, see `python run_exp.py --help`.

## Experimental results
Our experimental results can be downloaded from [here](https://storage.googleapis.com/iclr25_suppl/logs.zip). 
``` shell
> wget https://storage.googleapis.com/iclr25_suppl/logs.zip
```

Add the unzip version to the root directory of this project and see `python make_results.py --help` for reproducing plots and tables.

## Cite
Please cite ... if you use Stein mixture inference.
