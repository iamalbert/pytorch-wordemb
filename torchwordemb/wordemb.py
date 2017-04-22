import torch as _TH

from . import _torchwordemb

def load_glove_text(filename):

    tensor = _TH.FloatTensor()
    words = {}
    with open(filename) as f:
        p = _torchwordemb.load_glove(tensor, f.name.encode(), id(words) );

    return words, tensor

def load_word2vec_text(filename):

    tensor = _TH.FloatTensor()
    words = {}

    with open(filename) as f:
        p = _torchwordemb.load_word2vec(tensor, f.name.encode(), id(words) );

    return words, tensor

def load_word2vec_bin(filename):

    tensor = _TH.FloatTensor()
    words = {}

    with open(filename) as f:
        p = _torchwordemb.load_word2vec_bin(tensor, f.name.encode(), id(words) );

    return words, tensor

