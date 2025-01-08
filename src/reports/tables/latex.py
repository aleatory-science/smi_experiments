def state_better_same(eq, cond):
    if cond:
        return r"\bf{" + eq + "}"
    return eq

def state_better_same_vi(eq, cond):
    if cond:
        return r"\underline{" + eq + "}"
    return eq


def better(cond):
    if cond:
        return r"\surd"
    return r"\times"


def eq(eq: str):
    return rf"${eq}$"


def perf_mean(mean, std, prec: int, use_adjust):
    adjust = r"\negphantom{-}" if use_adjust and mean >= 0 else ""
    match prec:
        case 0:
            return rf"{adjust}{int(mean):,} \pm {int(std):,}"
        case 1:
            return rf"{adjust}{mean:,.1f} \pm {std:,.1f}"
        case 2:
            return rf"{adjust}{mean:,.2f} \pm {std:,.2f}"
        case 3:
            return rf"{adjust}{mean:,.3f} \pm {std:,.3f}"


def perf_median(median, prec: int):
    match prec:
        case 0:
            return rf"{int(median)}"
        case 1:
            return rf"{median:.1f}"
        case 2:
            return rf"{median:.2f}"
        case 3:
            return rf"{median:.3f}"
