import unittest
from ldamodel import TopicModel
from read import read_test
import numpy as np


class TestModelGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.model = None
        self.vocab_per_topic_size = 10
        self.num_topics = 10
        self.resources_path = "./test_resources"
        self.model = TopicModel(
            model_path="./testmodel/lda.model",
            dict_path="./testmodel/dictionary.dic",
            vocab_per_topic_size=self.vocab_per_topic_size,
            num_topics=self.num_topics,
        )

    def test_find_topic_for_word(self):
        test_token = "lady"
        expected_id = 17
        id = self.model.token2id(test_token)
        self.assertEqual(
            id,
            expected_id,
            f"Test token: '{test_token}' should have id: '{expected_id}' not '{id}'",
        )

    def test_top_words_for_topic(self):
        test_token = "lady"
        id = self.model.token2id(test_token)
        topic_id = self.model.word_topic_prob[:, id].argmax()
        top_word_ids = np.argsort(-self.model.word_topic_prob[topic_id, :])[
            : self.vocab_per_topic_size
        ]
        top_tokens = [self.model.id2token(top_id) for top_id in top_word_ids]
        old_topic_id = self.model._new_topic_to_old_topic[topic_id]
        top_tokens_in_topic = [
            token
            for token, prob in self.model.topic_model.show_topic(
                old_topic_id, topn=self.vocab_per_topic_size
            )
        ]
        self.assertListEqual(top_tokens, top_tokens_in_topic)

    def test_tau_calc(self):
        self.model.topic_model.random_state = np.random.RandomState(42)
        with open(f"{self.resources_path}/test_article.txt", "r") as f:
            test_document = f.read()
        test_tau = self.model.tau_calc(test_document.split())
        snapshot_tau = np.load(f"{self.resources_path}/snapshot_tau.npy")
        self.assertTrue(
            np.array_equal(test_tau, snapshot_tau),
            f"The tau calculation should equal the snapshot tau. dist: {np.linalg.norm(test_tau - snapshot_tau)}",
        )


if __name__ == "__main__":
    unittest.main()
