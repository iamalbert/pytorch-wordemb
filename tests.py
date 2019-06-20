import unittest

import torch
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

    def test_word2vec_bin(self):
        word, vec = torchwordemb.load_word2vec_bin("resource/word2vec.test.bin") 

        self.assertEqual(len(word), 113)

        self.assertEqual(vec.size(0), 113)
        self.assertEqual(vec.size(1), 100)


if __name__ == '__main__':
        print(dir(torchwordemb)) #.__path__)
        unittest.main()
#
