from ldamodel import TopicModel
from read import read_test
import numpy as np
import time

def get_token_weights(token_ids, token_weights, default_weight=1e-9):
    return [token_weights[0, token_id] if token_id != -1 else default_weight for token_id in token_ids]

model = TopicModel(model_path='./testmodel/lda.model', dict_path='./testmodel/dictionary.dic', vocab_size=10, num_topics=10)
model.get_word_topic_prob()
id = model._word_to_top_word[model.topic_model_dictionary.token2id['money']]
topic_id = model.get_word_topic_prob()[:, id].argmax()
print(model.topic_model.show_topic(model._top_topic_to_topic[topic_id]))
test_pd = read_test()
test_article = test_pd.article.iloc[0]
test_tau = model.tau_calc(test_article)
from pprint import pprint
pprint(model._word_topic_prob)
print(test_tau)