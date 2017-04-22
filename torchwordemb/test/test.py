import torch

import torchwordemb

def test_glove_text():
    vocab, tensor = torchwordemb.load_glove_text("glove.test.txt") 

    print(vocab)
    print(tensor)
    

if __name__ == '__main__':
    test_glove_text()
