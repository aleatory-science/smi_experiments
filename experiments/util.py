def total_exps(setup_exp):
    return sum(1 for _ in setup_exp())


def setup_logger(logger, log_type, config, set_config):
    match log_type:
        case "new":
            logger.write_exp_config(**config)
            logger.save_logs()
        case "latest":
            logger.load_latest_logs()
            logger.get_logs()
            # set_config(logger.load_exp_config())
            logger.write_exp_config(**config)
        case _:
            logger.load_logs(log_type)
            logger.get_logs()
            set_config(logger.load_exp_config())
