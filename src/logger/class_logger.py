"""This program define the experimental logger.

A logger captures and stores experimental measuresments and inferred models.

A logger consists of:

- write_entry
- write_artifact
- get_logs
- save_logs
- load_logs
- load_latest_logs
"""

import json
import dill
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from src.constants import EXPERIMENTS

import numpy as np


def _setup_dir(base, name):
    loc = base / name
    loc.mkdir(exist_ok=True, parents=True)
    return loc


def _timestamp():
    return str(datetime.now())


class ExpLogger:
    """Maintains, stores and loads experimental logs."""

    logs = {}
    ts = _timestamp()

    @staticmethod
    def to_dict(con):
        """Converts defaultdict(list) to dict for json serializing"""
        if isinstance(con, defaultdict):
            return dict(con)

        if isinstance(con, list):
            return con

        if isinstance(con, str):
            return con

        return {k: ExpLogger.to_dict(v) for k, v in con.items()}

    def __init__(self, experiment_name):
        assert experiment_name in EXPERIMENTS, "Unknown experiment."

        # Base directory for logs (same as root of project)
        logs_dir = _setup_dir(Path(__file__).parent.parent.parent, "logs")

        self.base_dir = _setup_dir(logs_dir, experiment_name)

        self.root = self.base_dir / self.ts

    def _add_entry(
        self,
        model_type,
        method,
        dataset,
    ):
        logs = self.logs
        if model_type not in logs:
            logs[model_type] = {}
        if method not in logs[model_type]:
            logs[model_type][method] = {}
        if dataset not in logs[model_type][method]:
            logs[model_type][method][dataset] = defaultdict(list)
        return logs

    def write_exp_config(self, **kwarg):
        fp = self.root
        fp.mkdir(exist_ok=True, parents=True)
        fname = "exp_config.json"

        with (fp / fname).open("w") as f:
            json.dump(kwarg, f)

        loc = str((fp / fname).relative_to(self.root))

        self.logs["exp_config"] = loc

    def load_exp_config(self):
        with (self.root / self.logs["exp_config"]).open() as f:
            return json.load(f)

    def merge(self, logger):
        other_logs = logger.get_logs()
        for mt, methods in other_logs.items():
            if mt == "exp_config":
                continue
            for method, res in methods.items():
                self.logs[mt][method] = res

    def write_artifact(
        self,
        model_type,
        method,
        dataset,
        guide,
        model,
        params,
        hyper_params,
        post_draws,
        summary,
    ):
        logs = self._add_entry(model_type, method, dataset)
        artifact = {
            "model": model,
            "guide": guide,
            "params": params,
            "hyper_params": hyper_params,
            "post_draws": post_draws,
            "summary": summary,
        }
        # For tables the logger save them locally and holds the location in the logger.
        fp = self.root / dataset / model_type / method / "artifact"
        fp.mkdir(exist_ok=True, parents=True)
        fname = f'{len(logs[model_type][method][dataset]['artifact'])}.dill'

        with open(fp / fname, "wb") as f:
            dill.dump(artifact, f)

        # Add location to logger
        loc = str((fp / fname).relative_to(self.root))

        logs[model_type][method][dataset]["artifact"].append(loc)

    def write_entry(self, model_type, method, dataset, **measures):
        logs = self._add_entry(model_type, method, dataset)

        for k, v in measures.items():
            if isinstance(v, list):
                # For tables the logger save them locally and holds the location in the logger.
                fp = self.root / dataset / model_type / method / k
                fp.mkdir(parents=True, exist_ok=True)
                fname = f"{len(logs[model_type][method][dataset][k])}.npy"
                np.save(fp / fname, np.array(v))

                # Add location to logger
                v = str((fp / fname).relative_to(self.root))

            logs[model_type][method][dataset][k].append(v)

    def get_logs(self):
        return ExpLogger.to_dict(self.logs)

    def save_logs(self):
        logs = self.logs
        fp = self.root
        fp.mkdir(exist_ok=True, parents=True)
        with (fp / "logs.json").open("w") as f:
            json.dump(ExpLogger.to_dict(logs), f)

    def load_logs(self, timestamp):
        self.timestamp = timestamp
        self.root = self.base_dir / self.timestamp

        fp = self.root / "logs.json"
        with fp.open("r") as f:
            self.logs = json.load(f)

    def load_latest_logs(self):
        ts = sorted(self.base_dir.glob("*"))
        self.load_logs(ts[-1])
