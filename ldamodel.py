from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import numpy as np
from collections import defaultdict
import torch
from sklearn.feature_extraction.text import CountVectorizer

# from https://github.com/ali-bahrainian/CATS/blob/main/data.py


class TopicModel(object):
    """Reads a pretrained LDA model trained using the Gensim library"""

    def __init__(
        self,
        model_path,
        dict_path,
        num_topics=None,
        vocab_per_topic_size=None,
        tokenizer=None,
        oov_value=1e-8,
    ):
        self.topic_model, self.topic_model_dictionary = self._load_pretrained_model(
            model_path, dict_path
        )
        self.topic_model.minimum_phi_value = 0.01
        self.topic_model.per_word_topics = False

        self.vocab_size = None
        self._word_topic_prob = None
        self.num_topics = (
            self.topic_model.num_topics if num_topics is None else num_topics
        )
        self._new_topic_to_old_topic = None
        self._old_topic_to_new_topic = None
        self._new_word_to_old_word = None
        self._old_word_to_new_word = None
        self.old_relevant_topics = None
        self.tokenizer = tokenizer
        self._oov_value = oov_value
        self.vocab_per_topic_size = vocab_per_topic_size

        self.word_topic_prob = self._gen_new_word_topic_prob()

    def _gen_new_word_topic_prob(self):
        print(f"Generating new word to topic probability matrix.")
        top_topics_with_words = self.topic_model.show_topics(
            num_topics=self.num_topics,
            num_words=self.vocab_per_topic_size,
            formatted=False,
        )
        top_topics = sorted([topic_id for topic_id, _ in top_topics_with_words])
        top_word_ids = {
            self.topic_model_dictionary.token2id[word]
            for _, word_list in top_topics_with_words
            for word, _ in word_list
        }
        top_word_ids = sorted(list(top_word_ids))
        self.vocab_size = len(top_word_ids)

        self._new_topic_to_old_topic = {i: topic for i, topic in enumerate(top_topics)}
        self._new_word_to_old_word = {i: word for i, word in enumerate(top_word_ids)}
        self._old_topic_to_new_topic = {
            topic: i for i, topic in self._new_topic_to_old_topic.items()
        }
        self._old_word_to_new_word = {word: i for i, word in self._new_word_to_old_word.items()}

        self.old_relevant_topics = sorted(list(self._old_topic_to_new_topic.keys()))

        all_topics_to_all_words = self.topic_model.get_topics()
        new_topics_to_new_words = all_topics_to_all_words[np.ix_(top_topics, top_word_ids)]
        return new_topics_to_new_words

    def _load_pretrained_model(self, model_path, dict_path):
        lda = LdaModel.load(model_path, mmap="r")
        print("Loaded the LDA model.")
        dictionary = Dictionary.load(dict_path, mmap="r")
        print("Loaded dictionary.")
        return lda, dictionary

    def doc2topics(self, tokens):
        vec_bow = self.topic_model_dictionary.doc2bow(tokens)
        perDocTopics = self.topic_model.get_document_topics(
            vec_bow, minimum_probability=0
        )
        return perDocTopics

    def token2id(self, token):
        return self._old_word_to_new_word.get(
            self.topic_model_dictionary.token2id.get(token, -1), -1
        )

    def id2token(self, id):
        return self.topic_model_dictionary.id2token.get(
            self._new_word_to_old_word.get(id, -1), -1
        )

    def get_token_weights(self, tokens):
        tau = self.tau_calc(tokens, with_oov = True)
        token_weights = [tau[self.token2id(token)] for token in tokens]
        return tokens, token_weights

    def sklearn_bow(self, articles):
        token_to_index_dict = {
            self.topic_model_dictionary.id2token[id]: index
            for id, index in self._old_word_to_new_word.items()
        }
        vectorizer = CountVectorizer(vocabulary=token_to_index_dict, tokenizer=self.tokenizer)
        return vectorizer.fit_transform(articles)

    def sklearn_bow_tensor(self, articles):
        return torch.tensor(self.sklearn_bow(articles).toarray())

    def doc_2_ids(self, doc):
        return [self.token2id.get(token, -1) for token in self.tokenizer(doc)]

    def tau_calc(self, tokens, with_oov=False):
        W = self.word_topic_prob
        T = np.mat(
            [
                topic_prob
                for _, topic_prob in self.doc2topics(tokens)
            ]
        )
        tau_d = T[:, self.old_relevant_topics] * W
        if with_oov:
            return np.append(tau_d, [[self._oov_value]], axis=1)
        else:
            return tau_d