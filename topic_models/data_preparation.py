import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from topic_models.tm_dataset import CTMDataset


def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]

    vect = scipy.sparse.csr_matrix(vect)
    return vect

class TopicModelDataPreparation:

    def __init__(self, vocab):
        self.vocab = vocab
        self.id2token = {}
        self.vectorizer = CountVectorizer(vocabulary=self.vocab)

    def create_training_set(self, text_for_bow):
        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)
        # self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        return CTMDataset(train_bow_embeddings, self.id2token)
