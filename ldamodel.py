from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import numpy as np
from collections import defaultdict 
import torch

# from https://github.com/ali-bahrainian/CATS/blob/main/data.py

class TopicModel(object):
    """Reads a pretrained LDA model trained using the Gensim library"""
    def __init__(self, model_path, dict_path, num_topics = None, vocab_per_topic_size=None, oov_value=1e-8):
        self.topic_model, self.topic_model_dictionary = self._load_pretrained_model(model_path, dict_path)
        self.topic_model.minimum_phi_value = 0.01
        self.topic_model.per_word_topics = False
        self.vocab_per_topic_size = vocab_per_topic_size
        self.vocab_size = None
        self._word_topic_prob = None
        self.num_topics = self.topic_model.num_topics if num_topics is None else num_topics
        self._top_topic_to_topic = None
        self._topic_to_top_topic = None
        self._top_word_to_word = None
        self._word_to_top_word = None
        self._oov_value = oov_value

    def get_word_topic_prob(self):
        if self._word_topic_prob is None:
            if self.vocab_per_topic_size is None:
                self._word_topic_prob = self.topic_model.get_topics()
            else:
                self._word_topic_prob = self._gen_new_word_topic_prob()
        return self._word_topic_prob
    
    def _gen_new_word_topic_prob(self):
        top_topics_with_words = self.topic_model.show_topics(num_topics = self.num_topics, num_words = self.vocab_per_topic_size, formatted=False)
        top_topics = sorted([topic_id for topic_id , _ in top_topics_with_words])
        top_words = {self.topic_model_dictionary.token2id[word] for _, word_list in top_topics_with_words for word, _ in word_list}
        top_words = sorted(list(top_words))
        self.vocab_size = len(top_words)
        self._top_topic_to_topic = {i: topic for i, topic in enumerate(top_topics)}
        self._top_word_to_word = {i: word for i, word in enumerate(top_words)}
        self._topic_to_top_topic = {topic: i for i, topic in self._top_topic_to_topic.items()}
        self._word_to_top_word = {word: i for i, word in self._top_word_to_word.items()}
        all_topics_to_all_words = self.topic_model.get_topics()
        return all_topics_to_all_words[np.ix_(top_topics, top_words)]
    
    def _load_pretrained_model (self, model_path, dict_path):
        lda = LdaModel.load(model_path, mmap = 'r')
        print("Loaded the LDA model.")
        dictionary = Dictionary.load(dict_path, mmap = 'r')
        print("Loaded dictionary.")
        return lda, dictionary
    
    def doc2topics(self, tokens):
        vec_bow = self.topic_model_dictionary.doc2bow(tokens)
        perDocTopics = self.topic_model.get_document_topics(vec_bow, minimum_probability=0)
        return perDocTopics

    def token2id(self, token):
        return self._word_to_top_word.get(self.topic_model_dictionary.token2id.get(token, -1), -1)

    def id2token(self, id):
        return self.topic_model_dictionary.id2token.get(self._top_word_to_word.get(id, -1), -1)

    def get_token_weights(self, tokens):
        tau = self.tau_calc(tokens)
        token_weights = [tau[self.token2id(token)] for token in tokens]
        return tokens, token_weights
    
    def bow_dict(self, tokens):
        bow_dict = defaultdict(lambda: 0)
        for token in tokens:
            bow_dict[self.token2id(token)] += 1
        return bow_dict
    
    def bow_tensor(self, tokens):
        bow_dict = self.bow_dict(tokens)
        bow_dict.pop(-1)
        word_indices = list(bow_dict.keys())
        word_counts =  list(bow_dict.values())
        indices_tensor = torch.tensor(word_indices)
        counts_tensor = torch.tensor(word_counts)
        return torch.sparse_coo_tensor(indices_tensor.view(1, -1), counts_tensor, size=(self.vocab_size,))

    def doc_2_ids(self, doc):
        return [self.token2id.get(token, -1) for token in doc.split()]

    def tau_calc(self, tokens):
        W = self.get_word_topic_prob()
        relevant_topics = set(self._topic_to_top_topic.keys())
        T = np.mat([topic_prob for topic_id, topic_prob in self.doc2topics(tokens) if topic_id in relevant_topics])
        tau_d = T*W
        return np.append(np.ravel(tau_d), self._oov_value)