# Stein Mixture Inference (SMI)
![Stein mixture inference uses a mixture model (with uniform weights) of $m$ guides $q(\theta|\psi_\ell)$, parameterized by particles $\psi_\ell$ to approximate $p(\theta|\mathcal{D})$. As a result, Stein mixture inference approximates a Bayesian posterior with a richer model that alleviates variance collapse in higher dimensional posteriors.](./images/smi.png)
SMI is a general-purpose particle-based inference algorithm for Bayesian modeling. Unlike other particle methods that transport point-like (volumeless) particles, SMI transports entire distributions. This distinction provides SMI particles with a notion of neighborhood, making SMI more robust to increasing model size.

SMI is available in [NumPyro](https://num.pyro.ai/en/stable/) as the [`SteinVI`](https://num.pyro.ai/en/stable/contrib.html#stein-variational-inference) inference engine. [`SteinVI`](https://num.pyro.ai/en/stable/contrib.html#stein-variational-inference) also supports [`SVGD`](https://arxiv.org/abs/1608.04471) and [`ASVGD`](https://arxiv.org/abs/2101.09815) as variants with their own constructors.

## Installation
This repo includes the experimental code for the article [ELBOing Stein: Variational Bayes with Stein Mixture Inference](https://arxiv.org/abs/2410.22948). 
To set up the project, ensure you have Python version `>=3.8`, `git` and the latest version of `pip`. You can check the version of Python and pip by
``` shell
> python --version
Python 3.12.6
> pip --version
pip 24.2 ...
```

To setup a virtual environment and install the requirements for the repo run 
``` shell
> python -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
```
*Please note that the experimental setup assumes a GPU device is available.*

If the [datasets](datasets) directory is empty, please use the following
``` shell
> git submodule add git@github.com:svendoc/datasets.git
> git submodule update --init --recursive
```

## Run experiments
To run the experiments, see `python run_exp.py --help` with the virtual environment activated.

## Experimental results
The experimental results from the article can be downloaded from [here](https://storage.googleapis.com/iclr25_suppl/logs.zip) or by commandline with `wget`
``` shell
> wget https://storage.googleapis.com/iclr25_suppl/logs.zip
```

Unzip `logs.zip` into to the base directory of this project, i.e. use 

``` shell
> unzip logs.zip 
```

Once the logs are unzipped use `python make_results.py --help` to reproduce plots and tables.

## Contact
If you’re interested in discussing SMI, exploring particle methods, or developing a SMI-based application, please reach out to me at **ronning`at`pm`dot`me**. If you need assistance with anything related to this repository, feel free to open an issue.

## Citation
Please use the following to cite the work
```
@article{ronning2024elboing,
  title={{ELBOing Stein: Variational Bayes with Stein Mixture Inference}},
  author={R{\o}nning, Ola and Nalisnick, Eric and Ley, Christophe and Smyth, Padhraic and Hamelryck, Thomas},
  journal={{arXiv preprint arXiv:2410.22948}},
  year={2024}
}
```