from math import log
import numpy as np

PARAM_K1 = 1.2
PARAM_B = 0.75
ALPHA_THRESHOLD = 0.9


# based on paper "Document Summarization Using Sentence-Level Semantic Based on Word Embeddings"

class Embedded_BM25:
    def __init__(self, sentences, dictionary, embedding_model):
        self.sentences = sentences
        self.dictionary = dictionary
        self.embedding_model = embedding_model
        self.avgSl = self._calculate_avgSl()
        self.numOfSentences = len(sentences)
        self.embidf = None
        self.emtf_dict = {}
        self._calculate_emidf_per_word()

    def _calculate_avgSl(self):
        len_sum = 0
        counter = 0
        for sentence in self.sentences:
            len_sum += len(sentence.token.split())
            counter += 1
        return len_sum / counter

    def _calculate_emidf_per_word(self):
        self.embidf = {}
        for term in self.dictionary.itervalues():
            n = 0
            s_index = 0
            self.emtf_dict[term] = {}
            for sentence in self.sentences:
                score = self._EMBTF(sentence.token.split(), term)
                self.emtf_dict[term][s_index] = score
                if score >= ALPHA_THRESHOLD:
                    n += 1
                s_index += 1
            self.embidf[term] = log(self.numOfSentences / (1 + n))

    def _EMBTF(self, Ss, w):
        max_cosine_similarity = None
        for word in Ss:
            similarity = self.embedding_model.similarity(word, w)
            if max_cosine_similarity is None:
                max_cosine_similarity = similarity
            elif similarity > max_cosine_similarity:
                max_cosine_similarity = similarity
        return max_cosine_similarity

    def _MBM25EMB(self, Sl_index, Ss_index):
        similarity = 0
        Sl = self.sentences[Sl_index].token.split()
        Ss = self.sentences[Ss_index].token.split()
        Ss_len = len(Ss)
        for word in Sl:
            embedding_tf = self.emtf_dict[word][Ss_index]
            embedding_idf = self.embidf[word]
            similarity += (embedding_idf * embedding_tf * (PARAM_K1 + 1)
                           / (embedding_tf + PARAM_K1 * (1 - PARAM_B + (PARAM_B * Ss_len / self.avgSl))))
        return similarity

    def get_similarity_matrix(self):
        size = len(self.sentences)
        sim_matrix = []
        for index1 in range(size - 1):
            sentence1_len = len(self.sentences[index1].token.split())
            for index2 in range(index1 + 1, size):
                sentence2_len = len(self.sentences[index2].token.split())
                long_sen_index = index1
                short_sen_index = index2
                if sentence1_len < sentence2_len:
                    tmp = long_sen_index
                    long_sen_index = short_sen_index
                    short_sen_index = tmp
                sim_matrix.append((index1, index2, self._MBM25EMB(long_sen_index, short_sen_index)))
        return sim_matrix
