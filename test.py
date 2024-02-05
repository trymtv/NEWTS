from ldamodel import TopicModel
from read import read_test
import numpy as np
import time

def get_token_weights(token_ids, token_weights, default_weight=1e-9):
    return [token_weights[0, token_id] if token_id != -1 else default_weight for token_id in token_ids]

model = TopicModel(model_path='./testmodel/lda.model', dict_path='./testmodel/dictionary.dic')
id = model.topic_model_dictionary.token2id['potato']
topic_id = model.get_word_topic_prob()[:, id].argmax()
print(model.topic_model.show_topic(topic_id))