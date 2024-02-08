from ldamodel import TopicModel
from read import read_test
import numpy as np
import time

def get_token_weights(token_ids, token_weights, default_weight=1e-9):
    return [token_weights[0, token_id] if token_id != -1 else default_weight for token_id in token_ids]

model = TopicModel(model_path='./testmodel/lda.model', dict_path='./testmodel/dictionary.dic', vocab_per_topic_size=1000, num_topics=100)
id = model.token2id('lady')
topic_id = model.word_topic_prob[:, id].argmax()
top_word_ids = np.argsort(-model.word_topic_prob[topic_id, :])[:10]
print([model.id2token(top_id) for top_id in top_word_ids])
print(model.topic_model.show_topic(model._new_topic_to_old_topic[topic_id]))
test_pd = read_test()
test_article = test_pd.article.iloc[0]
test_tokens = test_article.split()
test_tau, T = model.tau_calc(test_article.split())
new_test_tau, new_T = model.new_tau_calc(test_article.split())
print(test_tau == new_test_tau)
print(T == new_T)
print(T - new_T)
test1 = model.doc2topics(test_article.split()) 
test2 = model.doc2topics(test_article.split())
from pprint import pprint
#pprint(model._word_topic_prob)
#print(test_tau)
#print(model.bow_dict(test_tokens))
# tensor_numpy = model.bow_tensor(test_tokens).to_dense().numpy()
# sklearn_numpy = model.sklearn_bow([test_article]).toarray()
print('done')
#print(model.get_token_weights(test_tokens))