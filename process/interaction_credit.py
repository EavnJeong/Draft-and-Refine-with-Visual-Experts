"""Log-probability coalition credit. Keys are canonical subsets (EMPTY or ABCD)."""
from itertools import combinations


LABELS = 'ABCD'


def canonical(coalition):
    if coalition == 'EMPTY':
        return 'EMPTY'
    items = list(coalition)
    if len(items) != len(set(items)) or any(i not in LABELS for i in items):
        raise ValueError(f'Invalid coalition: {coalition}')
    return ''.join(sorted(items)) or 'EMPTY'


def subsets(coalition=LABELS):
    labels = canonical(coalition)
    labels = '' if labels == 'EMPTY' else labels
    return [canonical(c) for n in range(len(labels) + 1)
            for c in combinations(labels, n)]


def harsanyi_dividend(values, coalition):
    coalition = canonical(coalition)
    order = 0 if coalition == 'EMPTY' else len(coalition)
    return sum((-1) ** (order - (0 if t == 'EMPTY' else len(t))) * values[t]
               for t in subsets(coalition))


def standalone_marginal(values, i):
    if i not in LABELS or len(i) != 1:
        raise ValueError('Expected one evidence label')
    return values[i] - values['EMPTY']


def leave_one_out(values, i, full_set=LABELS):
    full_set = canonical(full_set)
    if len(i) != 1 or i not in full_set or full_set == 'EMPTY':
        raise ValueError('Label must belong to full_set')
    return values[full_set] - values[canonical(full_set.replace(i, ''))]


def pair_interaction(values, i, j):
    if i == j or len(i) != 1 or len(j) != 1:
        raise ValueError('Expected two distinct labels')
    return harsanyi_dividend(values, i + j)


def higher_order_interaction(values, coalition):
    return harsanyi_dividend(values, coalition)
