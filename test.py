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

    def test_bow(self):
        num_tokens = 10
        id_step = 5

        top_tokens = [
            self.model.id2token(self.model._old_word_to_new_word[id])
            for id in self.model.top_words_set
        ]

        wanted_tokens = top_tokens[: num_tokens * id_step : id_step]

        unwanted_tokens = [
            self.model.topic_model_dictionary.id2token[unwanted_id]
            for unwanted_id in range(
                self.model.vocab_size + 1, self.model.vocab_size + num_tokens
            )
        ]

        wanted_bow = {token: count + 1 for count, token in enumerate(wanted_tokens)}

        tokens = [
            token
            for wanted_token, count in wanted_bow.items()
            for token in [wanted_token] * count
        ]
        bow = self.model.bow(tokens + unwanted_tokens).toarray()

        not_null_indices = np.where((bow != 0))[1]
        null_indices = np.where((bow == 0))[1]

        token_bow = {
            self.model.id2token(id): count
            for id, count in zip(not_null_indices, bow[0][not_null_indices])
        }

        self.assertDictEqual(wanted_bow, token_bow)
        self.assertEqual(
            len(not_null_indices) + len(null_indices), self.model.vocab_size
        )


if __name__ == "__main__":
    unittest.main()
