from tqdm import tqdm


def product(params):
    """Produce all combinations of params"""

    def times(xs, ys):
        return [(*x, y) for y in ys for x in xs]

    vals = iter(params.values())
    prod = list((v,) for v in next(vals))

    for val in vals:
        prod = times(prod, val)

    keys = tuple(params.keys())
    for itm in prod:
        yield dict(zip(keys, itm))


def total(params):
    tot = 1
    for v in params.values():
        tot = tot * len(v)
    return tot


def grid_search(
    inf_key,
    eval_key,
    setup_engine,
    eval_fn,
    tr_args,
    tr_kwargs,
    val_args,
    search_space: dict,
):
    best = {"loss": float("inf"), "hparams": None}

    pbar = tqdm(product(search_space), total=total(search_space))

    for hparams in pbar:
        engine, run = setup_engine(**hparams)
        inf_res = run(inf_key, *tr_args, **tr_kwargs)
        loss = eval_fn(eval_key, val_args, engine, inf_res)
        if best["loss"] > loss:
            best = {"loss": loss, "hparams": hparams}
        pbar.set_description(f'best loss {best["loss"]:.3f} ({best["hparams"]})')

    return best["hparams"]
