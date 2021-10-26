# Quantum Computing Project 2021

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sntownsend/QuantumComputingProject2021/HEAD?labpath=%3Ftree)

This is a quantum NLP tutorial I wrote for COMP3710 Spring 2021 at ANU. It is based on the MS thesis of Thomas Hoffmann at Chalmers University, who kindly posted his code on [GitHub](https://github.com/Thommy257/Quantum-Models-for-Word-Sense-Disambiguation) and was very patient in helping me understand his work.

Investigating the application of compositional distributional models of meaning to a word-sense disambiguation task on predicate-argument relations.

The code can be executed on Binderhub using the link above. I've also provided a Dockerfile and a conda `environment.yml` file in the conda directory.

# Introduction to Quantum NLP

## Natural Language Processing

In the digital world, as we interact more and more with computers, it is increasingly necessary for computers to be able to understand our language. This is the job of the field of Natural Language Processing (NLP). Despite the fact that humans are so adept at acquiring and understanding human language, there are a variety of complicating factors making it more difficult to teach a computer to do the same.

Currently, the most popular statistical approaches to NLP are based on __Bag of Words__. In Bag-of-Words models of language, words are represented as one-hot vectors of dimension $N$, where $N$ is the number of words in your vocabulary. A document is then constructed by adding up all the vectors for the words in the document. Given a corpus consisting of a large number of documents ($M$), dimensionality reduction techniques such as SVD are used to reduce the vector space dimension to $N^\prime$ where $N^\prime << N$. For example, Google's popular Word2Vec algorithm has a vocabulary size of $N =  929022$, which has been reduced down to a vector of dimension 300. Word2Vec algorithms are effective and relatively easy to create, but a major limitation of Bag of Words is that the representation does not take into account any of the context of the words within a sentence or the larger document.

Newer statistical NLP models, such as BERT or GPT-3 model, model the context of words. However, this expanded representation comes at a huge cost in terms of the number of parameters that need to be fitted. For example, GPT-3 requires fitting nearly 200 billion (!) parameters.

Independent of these mainstream statistical NLP techniques, Joachim Lambek developed a mathematical grammar of language inspired by category theory and quantum mechanics. Lambek's grammar models lead directly to QNLP.

## Why QNLP? Word Sense Disambiguation

One of the reasons that statistical models such as GTP-3 require so many parameters, is that natural language is built up from words which are intrinsically ambiguous. Words usually have multiple different meanings. For example, the word "set" currently holds the Guinness World Record for English Word with the Most Meanings, with 430 senses listed in the 1989 Oxford English Dictionary. This is something humans can usually interpret with ease, but how do you teach a computer to do the same?

To take a simpler example, the word "file" has fourteen senses. Determining which sense should be used can sometimes be narrowed down. For example, if "file" is marked for tense or aspect, e.g. "filed" or "filing", this indicates that it can be restricted to just the verbal meanings, of which there are three more general and three more specific.


Take the sentences "She filed charges against you" and "He filed his teeth every day". We intuitively grasp the different meanings of the verbs in these sentences, despite the fact that they are pronounced and spelled the same, because of the extra information provided in the sentence, in this case the object of the verb. One job of natural language processing (NLP) is to teach a computer to do the same.


The ambiguity of words is precisely why quantum computing is a natural approach for NLP.

>[W]hat quantum theory and natural language share at a fundamental
level is an interaction structure. This interaction structure, together with the specification
of the spaces where the states live, determines the entire structure of processes of a theory.
So the fact that quantum theory and natural language also share the use of vector spaces
for describing states—albeit for very different reasons—makes those two theories (essentially)
coincide. Therefore we say that QNLP is ‘quantum-native’ (Coecke et al. p. 20)

In this tutorial I will illustrate how quantum computing can be used to disambiguate words. To do this, I will draw upon (with permission) the recent Master's Thesis of Thomas Hoffmann at Chalmers University.

## Workflow

As with most applications, a QNLP workflow will consist of a number of steps, some of which are done with classical computers and some of which would be done with quantum computers. Here I outline the basic workflow

- Tag each word in a sentence with its Part of Speech (PoS)
- Diagram each sentence according to the POS-tagging
- Diagram simplification
- Instantiate words as quantum features in circuit
- Translate each simplified diagram to a quantum circuit with words instantiated as quantum features.
- Compile and run each circuit on a quantum simulator.

### Each of these steps will be illustrated in the tutorial

### Part of Speech Tagging

Part-of-speech (POS) tagging is done using classical computing. A popular NLP package that does this quite well is the Python package [spaCy](spacy.io). For this demonstration, go to [this notebook](pos_tagging.ipynb).

### Generating Sentence Diagrams

Click on [this notebook](SentenceDiagramming.ipynb) to see how to generate sentence diagrams.

```Python
sentences = [
    "file account",
    "register account",
    "smooth account",
    "file nail",
    "register nail",
    "smooth nail",
    "file charge_n",
```

### Diagram Simplification

- Graphical Category Languages based on quantum physics

### [Word Vectors](dimensionalty_reduction.ipynb)

- [wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/)
creates word vectors based on Wikipedia data dumps using a method similar to that described for Word2Vec above, but with a resulting dimensionality of 100.

For example, the word vector for "account" is

```Python
account = \
[ 0.26, -0.06,  0.01,  0.35, -0.03,  0.04,  0.16,  0.14,  0.4 ,
        0.02,  0.49, -0.19,  0.17,  0.2 ,  0.46,  0.12, -0.07,  0.07,
        0.33, -0.31,  0.23, -0.35,  0.03,  0.51, -0.46, -0.16, -0.55,
       -0.48,  0.4 ,  0.35, -0.02, -0.09,  0.34,  0.71, -0.16, -0.22,
        0.23,  0.42,  0.09,  0.23, -0.1 , -0.23, -0.5 ,  0.32,  0.07,
        0.25, -0.19,  0.33, -0.27,  0.29,  0.13, -0.06,  0.17,  0.58,
        0.12,  0.05, -0.17,  0.17,  0.  ,  0.2 , -0.19, -0.25,  0.07,
        0.26,  0.59, -0.65, -0.33, -0.31, -0.31, -0.39,  0.05,  0.36,
        0.13,  0.31, -0.27,  0.51, -0.12,  0.29, -0.06, -0.3 , -0.41,
       -0.48,  0.26, -0.01,  0.36,  0.63, -0.09, -0.15,  0.19,  0.09,
        0.28,  0.15,  0.24, -0.04, -0.38, -0.63, -0.  ,  0.04, -0.25,
        0.2 ]
```

The word vector for each 35 words in our vocabulary are computed and used to generate a $35\times 100$ matrix. [Dimensionality reduction](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) is then used to reduce this matrix to a $35\times 4$ matrix.

In this reduced dimensional space, "account" is now represented by the vector:

```Python
[-0.92, -0.29, -0.023, 0.28]

```

### Key components

- Word representation
- Density Matrices

### Example 1-Qubit Model

- Two state verbs

### Optimization


```python

```


## Noisyopt

The noisyopt package was adapted from

https://github.com/andim/noisyopt/blob/master/doc/index.rst
