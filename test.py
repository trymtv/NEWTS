from ldamodel import TopicModel
from read import read_test
import numpy as np
import time

def time_tau_calc(topic_model, article, time=False):
    W = topic_model.get_word_topic_prob()
    T = np.mat([topic_prob for topic_id, topic_prob in topic_model.doc2topics(article)])
    tau_d = T*W
    return tau_d

model = TopicModel(modelAdd='./testmodel/lda.model', dictAdd='./testmodel/dictionary.dic')
test_pd = read_test()
all_tau = [time_tau_calc(model, article) for article in test_pd.article]
print(np.concatenate(all_tau))