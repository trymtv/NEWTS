from ldamodel import TopicModel
from read import read_test
import numpy as np
import time

def get_token_weights(token_ids, token_weights, default_weight=1e-9):
    return [token_weights[0, token_id] if token_id != -1 else default_weight for token_id in token_ids]

model = TopicModel(model_path='./testmodel/lda.model', dict_path='./testmodel/dictionary.dic', vocab_per_topic_size=1000, num_topics=100)
model.get_word_topic_prob()
id = model.token2id('lady')
topic_id = model.get_word_topic_prob()[:, id].argmax()
top_word_ids = np.argsort(-model.get_word_topic_prob()[topic_id, :])[:10]
print([model.id2token(top_id) for top_id in top_word_ids])
print(model.topic_model.show_topic(model._top_topic_to_topic[topic_id]))
test_pd = read_test()
test_article = test_pd.article.iloc[0]
test_tokens = test_article.split()
test_tau = model.tau_calc(test_article.split())
from pprint import pprint
#pprint(model._word_topic_prob)
#print(test_tau)
#print(model.bow_dict(test_tokens))
tensor_numpy = model.bow_tensor(test_tokens).to_dense().numpy()
sklearn_numpy = model.sklearn_bow([test_article]).toarray()
print('done')
#print(model.get_token_weights(test_tokens))