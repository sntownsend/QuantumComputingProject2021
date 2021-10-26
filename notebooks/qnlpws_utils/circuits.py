from discopy.rigid import Ty, Diagram, Id, Cup

from discopy.quantum.gates import Ket, H, Rx, Ry, Rz, CX, sqrt, X
from discopy.quantum.circuit import Circuit
from discopy import CircuitFunctor, qubit
from random import uniform

from . import Vocab, Params

s, n = Ty('s'), Ty('n')

# prepare intitial params

# Ansätze for 1-qubit states
def un_amb_verb_ansatz(p):
    return Ket(p[0])

def amb_verb_ansatz(p):
    return Ket(0) >>  \
        Rx(p[0])

def noun_ansatz(p):
    return Ket(0) >> \
        Rx(p[0]) >> \
        Rz(p[1])

def create_params(nouns, 
                  ambiguous_verbs, 
                  vocab,
                  n_qubits_ansatz = 1,
                  n_noun_params = 2,
                  n_amb_params = 1):
    binaries = [list(bin(i)[2:]) for i, verb in enumerate(vocab.unamb_verb.keys())]
    int_binaries = [[0]*(n_qubits_ansatz - len(bi)) + [int(b) for b in bi] for bi in binaries]

    n_nouns = len(nouns)
    n_amb_verbs = len(ambiguous_verbs)
    n_unamb_verbs = len(vocab.unamb_verb)

    params_unamb_verbs = {verb: int_binaries[i] for i, verb in enumerate(vocab.unamb_verb.keys())}
    params_nouns = {noun: [uniform(0, 1) for i in range(n_noun_params)] for noun in vocab.noun}
    params_amb_verbs = {verb: [uniform(0, 1) for i in range(n_amb_params)] for verb in vocab.amb_verb}

    return Vocab(params_nouns, params_unamb_verbs, params_amb_verbs)

def F(vocab,
      params,
      n_qubits_ansatz=1):
    ar1 = {vocab.noun[noun]:noun_ansatz(params.noun[noun]) for noun in vocab.noun}
    ar2 = {vocab.unamb_verb[verb]:un_amb_verb_ansatz(params.unamb_verb[verb]) for verb in vocab.unamb_verb}
    ar3 = {vocab.amb_verb[verb]:amb_verb_ansatz(params.amb_verb[verb]) for verb in vocab.amb_verb}
    ar = {**ar1, **ar2, **ar3}

    return CircuitFunctor(
        ob = {s: qubit ** 0, n: qubit ** n_qubits_ansatz},
        ar = ar)

