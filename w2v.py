import pickle
import gensim
import time
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

class WordEmbeddingModel:
    def __init__(self, model_path=os.path.join(dir_path, 'nlp_data/glove.6B.100d.w2vformat.txt'), cache=True):
        self.model = self._load_model(model_path, cache)

    def _load_model(self,path, cache):
        start = time.time()
        w2v_model_path = os.path.join(dir_path, 'nlp_data/w2vmodel')
        if cache is True and os.path.exists(w2v_model_path):
            print('Loading W2V model from cache')
            model = pickle.load(open(w2v_model_path, 'rb'))
        else:
            print('Generating W2V model')
            model = gensim.models.KeyedVectors.load_word2vec_format(path)
            pickle.dump(model, open(w2v_model_path, 'wb'))
        print("***W2V model loading--" + str((time.time() - start) / 60) + ' minutes')
        return model


if __name__ == '__main__':
    pass
