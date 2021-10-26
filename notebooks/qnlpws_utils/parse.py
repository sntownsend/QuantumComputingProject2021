from . import Vocab, Params


def parse_dummy_sentences(sentences):
    dataset = []
    nouns, verbs = set(), set()
    for s, p in sentences.items():
        _s = s.split(" ")
        dataset.append([_s[0],_s[1],p])
        verbs.add(_s[0])
        nouns.add(_s[1])
    return dataset, nouns, verbs

def parse_with_grammar(dataset, grammar, vocab):

    sentences, plausability, parsing = list(), list(), dict()

    for entry in dataset:
        sentence = ' '.join(entry[0:2]) + '.'
        sentences.append(sentence)
        plausability.append(entry[2])

        verb = vocab.unamb_verb.get(entry[0], False)
        if not verb:
            verb = vocab.amb_verb[entry[0]]
        obj = vocab.noun[entry[1]]
        diagram = verb @ obj >> grammar

        parsing.update({sentence: diagram})
    return sentences, plausability, parsing
