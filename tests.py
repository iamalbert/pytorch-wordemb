import unittest

import torchwordemb


class TestGlove(unittest.TestCase):
    def test_glove_text(self):
        word, vec = torchwordemb.load_glove_text("resource/glove.test.txt")

        self.assertEqual(len(word), 10)

        self.assertEqual(vec.size(0), 10)
        self.assertEqual(vec.size(1), 300)

    def test_word2vec_text(self):
        word, vec = torchwordemb.load_word2vec_text("resource/word2vec.test.txt")

        self.assertEqual(len(word), 10)

        self.assertEqual(vec.size(0), 10)
        self.assertEqual(vec.size(1), 300)
