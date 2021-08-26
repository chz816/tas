"""
This file is modified from the "contextualized_topic_models" package - preprocessing.py
"""
import os

from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords as stop_words


class WhiteSpacePreprocessing():
    """
    Provides a very simple preprocessing script that filters infrequent tokens from text
    """

    def __init__(self, documents, stopwords_language="english", vocabulary_size=2000):
        """

        :param documents: list of strings
        :param stopwords_language: string of the language of the stopwords (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.stopwords = set(stop_words.words(stopwords_language))
        self.vocabulary_size = vocabulary_size

    def preprocess(self, vocabulary=None, save=False, save_path=None):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        # remove the punctuation
        preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc
                                 in preprocessed_docs_tmp]

        if vocabulary is None:
            # learn the vocabulary
            vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
            vectorizer.fit_transform(preprocessed_docs_tmp)
            vocabulary = set(vectorizer.get_feature_names())

        preprocessed_docs = []
        for i, doc in enumerate(preprocessed_docs_tmp):
            processed_doc = ' '.join([w for w in doc.split() if len(
                w) > 0 and w not in self.stopwords and not w.isdigit() and w in vocabulary])
            if len(processed_doc) > 0:
                preprocessed_docs.append(processed_doc)
            else:
                preprocessed_docs.append(preprocessed_docs[-1])

        # save the preprocessed_docs
        if save and save_path is not None:
            with open(save_path, "w") as f:
                for line in preprocessed_docs:
                    f.write(f"{line}\n")

        return preprocessed_docs, list(vocabulary)
