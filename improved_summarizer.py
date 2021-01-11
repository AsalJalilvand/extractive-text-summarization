#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module provides functions for summarizing texts. Summarizing is based on
ranks of text sentences using a variation of the TextRank algorithm [1]_.

.. [1] Federico Barrios, Federico L´opez, Luis Argerich, Rosita Wachenchauzer (2016).
       Variations of the Similarity Function of TextRank for Automated Summarization,
       https://arxiv.org/abs/1602.03606


Data
----

.. data:: INPUT_MIN_LENGTH - Minimal number of sentences in text
.. data:: WEIGHT_THRESHOLD - Minimal weight of edge between graph nodes. Smaller weights set to zero.

Example
-------

.. sourcecode:: pycon
"""

import logging
from gensim.utils import deprecated
from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.commons import build_graph as _build_graph
from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from gensim.summarization.bm25 import iter_bm25_bow as _bm25_weights
from gensim.corpora import Dictionary
from math import log10 as _log10
from six.moves import range

# ---Asal
from modified_bm25 import Embedded_BM25
from textcleaner_without_stemming import clean_text_by_sentences as _clean_text_by_sentences
from nltk.corpus import stopwords
from gensim.summarization.summarizer import summarize as gensim_original_summarize

INPUT_MIN_LENGTH = 10

WEIGHT_THRESHOLD = 1.e-3

REDUNDANCY_THRESHOLD = 0.8
QUERY_RELEVANCE_THRESHOLD = 0.5

logger = logging.getLogger(__name__)

embedding_model = None


def _set_graph_edge_weights(graph, dictionary):
    """Sets weights using BM25 algorithm. Leaves small weights as zeroes. If all weights are fairly small,
     forces all weights to 1, inplace.

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.

    """
    documents = graph.nodes()
    embm25 = Embedded_BM25(documents, dictionary, embedding_model)
    weights = embm25.get_similarity_matrix()
    for w in weights:
        weight = w[2]
        if weight < WEIGHT_THRESHOLD:
            continue
        edge = (documents[w[0]], documents[w[1]])

        if not graph.has_edge(edge):
            graph.add_edge(edge, weight)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.iter_edges()):
        _create_valid_graph(graph)


def _create_valid_graph(graph):
    """Sets all weights of edges for different edges as 1, inplace.

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.

    """
    nodes = graph.nodes()

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


@deprecated("Function will be removed in 4.0.0")
def _get_doc_length(doc):
    """Get length of (tokenized) document.

    Parameters
    ----------
    doc : list of (list of (tuple of int))
        Given document.

    Returns
    -------
    int
        Length of document.

    """
    return sum(item[1] for item in doc)


@deprecated("Function will be removed in 4.0.0")
def _get_similarity(doc1, doc2, vec1, vec2):
    """Returns similarity of two documents.

    Parameters
    ----------
    doc1 : list of (list of (tuple of int))
        First document.
    doc2 : list of (list of (tuple of int))
        Second document.
    vec1 : array
        ? of first document.
    vec1 : array
        ? of secont document.

    Returns
    -------
    float
        Similarity of two documents.

    """
    numerator = vec1.dot(vec2.transpose()).toarray()[0][0]
    length_1 = _get_doc_length(doc1)
    length_2 = _get_doc_length(doc2)

    denominator = _log10(length_1) + _log10(length_2) if length_1 > 0 and length_2 > 0 else 0

    return numerator / denominator if denominator != 0 else 0


def _build_corpus(sentences):
    """Construct corpus from provided sentences.

    Parameters
    ----------
    sentences : list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Given sentences.

    Returns
    -------
    list of list of (int, int)
        Corpus built from sentences.

    """
    split_tokens = [sentence.token.split() for sentence in sentences]
    dictionary = Dictionary(split_tokens)
    return [dictionary.doc2bow(token) for token in split_tokens], dictionary


def _get_important_sentences(sentences, corpus, important_docs):
    """Get most important sentences.

    Parameters
    ----------
    sentences : list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Given sentences.
    corpus : list of list of (int, int)
        Provided corpus.
    important_docs : list of list of (int, int)
        Most important documents of the corpus.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Most important sentences.

    """
    hashable_corpus = _build_hasheable_corpus(corpus)
    sentences_by_corpus = dict(zip(hashable_corpus, sentences))
    return [sentences_by_corpus[tuple(important_doc)] for important_doc in important_docs]


def _get_sentences_with_word_count(sentences, word_count):
    """Get list of sentences. Total number of returned words close to specified `word_count`.

    Parameters
    ----------
    sentences : list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Given sentences.
    word_count : int or None
        Number of returned words. If None full most important sentences will be returned.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Most important sentences.

    """
    length = 0
    selected_sentences = []

    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = sentence.text.split()
        length_of_sentence = len(words_in_sentence)

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(word_count - length - length_of_sentence) > abs(word_count - length):
            return selected_sentences
        selected_sentences.append(sentence)
        length += length_of_sentence

    return selected_sentences


def _is_redundant(words_in_sentence, summary):
    if len(summary) == 0:
        return False
    sum = 0
    for s in summary:
        sum += embedding_model.n_similarity(s.token.split(), words_in_sentence)
    return sum / len(summary) > REDUNDANCY_THRESHOLD

def _extract_important_sentences(important_sentences, word_count):
    """Get most important sentences of the `corpus`.

    Parameters
    ----------
    sentences : list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Given sentences.
    corpus : list of list of (int, int)
        Provided corpus.
    important_docs : list of list of (int, int)
        Most important docs of the corpus.
    word_count : int
        Number of returned words. If None full most important sentences will be returned.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Most important sentences.

    """

    # If no "word_count" option is provided, the number of sentences is
    # reduced by the provided ratio. Else, the ratio is ignored.
    return important_sentences \
        if word_count is None \
        else _get_sentences_with_word_count(important_sentences, word_count)


def _format_results(extracted_sentences, split):
    """Returns `extracted_sentences` in desired format.

    Parameters
    ----------
    extracted_sentences : list of :class:~gensim.summarization.syntactic_unit.SyntacticUnit
        Given sentences.
    split : bool
        If True sentences will be returned as list. Otherwise sentences will be merged and returned as string.

    Returns
    -------
    list of str
        If `split` **OR**
    str
        Formatted result.

    """
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join(sentence.text for sentence in extracted_sentences)


def _build_hasheable_corpus(corpus):
    """Hashes and get `corpus`.

    Parameters
    ----------
    corpus : list of list of (int, int)
        Given corpus.

    Returns
    -------
    list of list of (int, int)
        Hashable corpus.

    """
    return [tuple(doc) for doc in corpus]


def summarize_corpus(corpus, dictionary, sentences, ratio=0.2, redundancy_check=True, query=None):
    """Get a list of the most important documents of a corpus using a variation of the TextRank algorithm [1]_.
     Used as helper for summarize :func:`~gensim.summarization.summarizer.summarizer`

    Note
    ----
    The input must have at least :const:`~gensim.summarization.summarizer.INPUT_MIN_LENGTH` documents for the summary
    to make sense.


    Parameters
    ----------
    corpus : list of list of (int, int)
        Given corpus.
    ratio : float, optional
        Number between 0 and 1 that determines the proportion of the number of
        sentences of the original text to be chosen for the summary, optional.

    Returns
    -------
    list of str
        Most important documents of given `corpus` sorted by the document score, highest first.

    """
    #hashable_corpus = _build_hasheable_corpus(corpus)

    # If the corpus is empty, the function ends.
    '''if len(corpus) == 0:
        logger.warning("Input corpus is empty.")
        return []

    # Warns the user if there are too few documents.
    if len(corpus) < INPUT_MIN_LENGTH:
        logger.warning("Input corpus is expected to have at least %d documents.", INPUT_MIN_LENGTH)'''

    logger.info('Building graph')
    graph = _build_graph(sentences)

    logger.info('Filling graph')
    _set_graph_edge_weights(graph, dictionary)

    logger.info('Removing unreachable nodes of graph')
    _remove_unreachable_nodes(graph)

    # Cannot calculate eigenvectors if number of unique documents in corpus < 3.
    # Warns user to add more text. The function ends.
    if len(graph.nodes()) < 3:
        logger.warning("Please add more sentences to the text. The number of reachable nodes is below 3")
        return []

    logger.info('Pagerank graph')
    pagerank_scores = _pagerank(graph)

    logger.info('Sorting pagerank scores')
    sentences.sort(key=lambda doc: pagerank_scores.get(doc, 0), reverse=True)

    if redundancy_check or (query is not None):
        selected = []
        counter = 0
        while (len(selected) <= int(len(corpus) * ratio) and counter < len(corpus)):
            sentence_words = sentences[counter].token.split()
            if redundancy_check and _is_redundant(sentence_words, selected):
                counter += 1
                continue
            if _is_related_to_query(sentence_words, query):
                selected.append(sentences[counter])
            counter += 1
        return selected

    return sentences[:int(len(corpus) * ratio)]


def _is_related_to_query(sentence_words, query_list):
    if query_list is not None:
        return embedding_model.n_similarity(sentence_words, query_list) >= QUERY_RELEVANCE_THRESHOLD
    return True


def summarize(text, ratio=0.2, word_count=None, query=None, split=False, redundancy_removal=True):
    """Get a summarized version of the given text.

    The output summary will consist of the most representative sentences
    and will be returned as a string, divided by newlines.

    Note
    ----
    The input should be a string, and must be longer than :const:`~gensim.summarization.summarizer.INPUT_MIN_LENGTH`
    sentences for the summary to make sense.
    The text will be split into sentences using the split_sentences method in the :mod:`gensim.summarization.texcleaner`
    module. Note that newlines divide sentences.


    Parameters
    ----------
    text : str
        Given text.
    ratio : float, optional
        Number between 0 and 1 that determines the proportion of the number of
        sentences of the original text to be chosen for the summary.
    word_count : int or None, optional
        Determines how many words will the output contain.
        If both parameters are provided, the ratio will be ignored.
    split : bool, optional
        If True, list of sentences will be returned. Otherwise joined
        strings will bwe returned.

    Returns
    -------
    list of str
        If `split` **OR**
    str
        Most representative sentences of given the text.

    """
    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text)
    remove_list = []
    for sentence in sentences:
        s = sentence.token.split()
        available = get_available_terms_in_model(s)
        if len(available) != 0:
            sentence.token = ' '.join(available)
        else:
            remove_list.append(sentence)
    for r in remove_list:
        sentences.remove(r)

    # If no sentence could be identified, the function ends.
    if len(sentences) == 0:
        logger.warning("Input text is empty.")
        return [] if split else u""

    # If only one sentence is present, the function raises an error (Avoids ZeroDivisionError).
    if len(sentences) == 1:
        raise ValueError("input must have more than one sentence")

    # Warns if the text is too short.
    '''if len(sentences) < INPUT_MIN_LENGTH:
        logger.warning("Input text is expected to have at least %d sentences.", INPUT_MIN_LENGTH)'''

    corpus, dictionary = _build_corpus(sentences)

    if query is not None:
        query = [q.lower() for q in query]
        query = [word for word in query if not word in stopwords.words()]
        query = get_available_terms_in_model(query)
        if len(query) == 0:
            query = None  # simple summarization if query tokens not available in w2v model

    most_important_docs = summarize_corpus(corpus, dictionary, sentences, ratio=ratio if word_count is None else 1,
                                           redundancy_check=redundancy_removal, query=query)
    # If couldn't get important docs, the algorithm ends.
    if not most_important_docs:
        logger.warning("Couldn't get relevant sentences.")
        return [] if split else u""

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_important_sentences(most_important_docs, word_count)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split)


def set_embedding_model(model):
    global embedding_model
    embedding_model = model


def get_available_terms_in_model(list):
    available = []
    for term in list:
        if term in embedding_model.vocab:
            available.append(term)
    return available


def query_based_summarization_baseline(text, ratio, query):
    summary = gensim_original_summarize(text, ratio=ratio, split=True)
    query_related_sentences = set()
    for i in range(len(summary)):
        for q in query:
            if q in summary[i]:
                query_related_sentences.add(i)
                continue
    result = []
    for i in query_related_sentences:
        result.append(summary[i])
    return ' '.join(result)
