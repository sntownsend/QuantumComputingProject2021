from lambeq import AtomicType, Sim14Ansatz, IQPAnsatz

from . import Vocab  # noqa: F401  (re-exported for notebook imports)

s = AtomicType.SENTENCE
n = AtomicType.NOUN


def make_ansatz(n_layers=1, ansatz_type='sim14'):
    """Create a circuit ansatz mapping nouns to 1-qubit states, sentences to scalars.

    Replaces the old CircuitFunctor + per-word ansatz functions (noun_ansatz,
    amb_verb_ansatz, un_amb_verb_ansatz). Parameters are now managed by the
    NumpyModel / TketModel rather than a separate Params namedtuple.

    ansatz_type: 'sim14' (Ry + CRx, default) or 'iqp' (H + CRz)
    n_layers: number of ansatz repetition layers
    """
    ob_map = {n: 1, s: 0}
    if ansatz_type == 'iqp':
        return IQPAnsatz(ob_map, n_layers=n_layers)
    return Sim14Ansatz(ob_map, n_layers=n_layers)
