import gensim
from gensim import corpora

# from https://github.com/ali-bahrainian/CATS/blob/main/data.py

class TopicModel(object):
    """Reads a pretrained LDA model trained using the Gensim library"""
    def __init__(self, modelAdd, dictAdd):
        self.topicModel, self.topicModelDictionary = self._loadPretrainedTM(modelAdd, dictAdd)
        self.topicModel.minimum_phi_value = 0.01
        self.topicModel.per_word_topics = False
        self.word_topic_prob = None

    def get_word_topic_prob(self):
        if self.word_topic_prob is None:
            self.word_topic_prob = self.topicModel.get_topics()
        return self.word_topic_prob
    
    def _loadPretrainedTM (self, modelAdd, dictAdd):
        lda = gensim.models.ldamodel.LdaModel.load(modelAdd, mmap = 'r')
        print("Loaded the LDA model.")
        dictionary = corpora.Dictionary.load(dictAdd, mmap = 'r')
        print("Loaded dictionary.")
        return lda, dictionary
    
    def doc2topics (self, doc):
        vec_bow = self.topicModelDictionary.doc2bow(doc.split())
        perDocTopics = self.topicModel.get_document_topics(vec_bow, minimum_probability=0)
        return perDocTopics