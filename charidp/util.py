import tarfile

from keras import backend as K
import theano
from theano import tensor as T
from keras.layers import GRU, LSTM, merge
from keras.engine.topology import InputSpec
import random

import logging


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)


def build_model_args(config, vectorizer, hierarchical=False):
    model_args = {
        "hierarchical": False,
        "seq2seq_model": config["seq2seq_model"],
        "num_inputs": vectorizer.vocab_size,
        "answer_maxlen": vectorizer.answer_maxlen,
        "query_maxlen": vectorizer.query_maxlen,
        "conv_nfilters": config["conv_nfilters"],
        "query_lstm_dims": config["query_lstm_dims"],
        "hierarchical_lstm_dims": config["hierarchical_lstm_dims"],
        "dropout": config["dropout"],
        "char_embedding_size": config["char_embedding_size"],
        "hidden_dim": config["hidden_dim"],
        "encoder_depth": config["encoder_depth"],
        "decoder_depth": config["decoder_depth"],
        "seq_maxlen": vectorizer.sent_maxlen,
        "story_maxlen": None
    }
    if hierarchical:
        model_args["seq_maxlen"] = vectorizer.sent_maxlen
        model_args["story_maxlen"] = vectorizer.story_maxlen
    else:
        model_args["seq_maxlen"] = vectorizer.sent_maxlen
        model_args["story_maxlen"] = None

    return model_args


def get_logger(name):
    return logging.getLogger(name)


def BiDirectional(Recurrent_, return_sequences=True, name=None):
    def _bidirectional(layer):
        encoder_forward = Recurrent_(
            256, return_sequences=return_sequences, dropout_W=0.3, dropout_U=0.3, name=name + "_forward"
        )(layer)

        encoder_backward = Recurrent_(
            256, return_sequences=return_sequences, go_backwards=True, dropout_W=0.3, dropout_U=0.3, name=name + "_backward"
        )(layer)

        return merge([encoder_forward, encoder_backward], mode="concat")

    return _bidirectional


class ShareableGRU(GRU):
    def __init__(self, *args, **kwargs):
        super(ShareableGRU, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        res = super(ShareableGRU, self).__call__(*args, **kwargs)
        self.input_spec = [InputSpec(shape=(self.get_input_shape_at(0)[0], None, self.get_input_shape_at(0)[2]))]
        return res


class ShareableLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        super(ShareableLSTM, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        res = super(ShareableLSTM, self).__call__(*args, **kwargs)
        self.input_spec = [InputSpec(shape=(self.get_input_shape_at(0)[0], None, self.get_input_shape_at(0)[2]))]
        return res


def kullback_leibler(y_pred, y_true):
    # from https://github.com/fchollet/keras/issues/2473
    eps = 0.0001
    results, updates = theano.scan(
        lambda y_t, y_p: (y_t + eps) * (T.log(y_t + eps) - T.log(y_p + eps)), sequences=[y_true, y_pred]
    )
    return T.sum(results, axis=-1)


def shuffle_together(matrices):
    num_samples = matrices[0].shape[0]
    indices = range(num_samples)
    random.shuffle(indices)
    res = []
    for matrix in matrices:
        res.append(matrix[indices])
    return res


