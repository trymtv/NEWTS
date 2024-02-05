from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import numpy as np

# from https://github.com/ali-bahrainian/CATS/blob/main/data.py

class TopicModel(object):
    """Reads a pretrained LDA model trained using the Gensim library"""
    def __init__(self, model_path, dict_path):
        self.topic_model, self.topic_model_dictionary = self._load_pretrained_model(model_path, dict_path)
        self.topic_model.minimum_phi_value = 0.01
        self.topic_model.per_word_topics = False
        self.word_topic_prob = None

    def get_word_topic_prob(self):
        if self.word_topic_prob is None:
            self.word_topic_prob = self.topic_model.get_topics()
        return self.word_topic_prob
    
    def _load_pretrained_model (self, model_path, dict_path):
        lda = LdaModel.load(model_path, mmap = 'r')
        print("Loaded the LDA model.")
        dictionary = Dictionary.load(dict_path, mmap = 'r')
        print("Loaded dictionary.")
        return lda, dictionary
    
    def doc2topics (self, doc):
        vec_bow = self.topic_model_dictionary.doc2bow(doc.split())
        perDocTopics = self.topic_model.get_document_topics(vec_bow, minimum_probability=0)
        return perDocTopics

    def doc_2_ids(self, doc):
        return [self.topic_model_dictionary.token2id.get(token, -1) for token in doc.split()]

    def tau_calc(self, article, time=False):
        W = self.get_word_topic_prob()
        T = np.mat([topic_prob for topic_id, topic_prob in self.doc2topics(article)])
        tau_d = T*W
        return tau_d