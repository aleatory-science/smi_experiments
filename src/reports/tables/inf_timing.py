from src.logger import ExpLogger
import numpy as np

MT = "bnn"


def pprint_timing(logger):
    logs = logger.get_logs()

    for method, value in logs[MT].items():
        kernel = list(value.keys())[0]
        for dataset, mesures in value[kernel].items():
            inf_time = np.array(mesures["inf_time"])
            print(method, dataset, int(inf_time.mean()))


logger = ExpLogger("uci_std")
logger.load_logs("2024-11-15 06:17:49.793897")
pprint_timing(logger)
