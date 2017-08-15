from keras.models import Model as _Model
from keras.layers import Dense, Input, Dropout, Embedding, TimeDistributed,  Convolution1D, MaxPooling1D, LSTM, \
    RepeatVector
try:
    from keras.layers.merge import concatenate, add
    Model = _Model
except ImportError:
    print "Seems to be old keras version"
    from keras.layers import merge
    concatenate = lambda inp: merge(inp, mode="concat")
    add = lambda inp: merge(inp, mode="add")
    Model = lambda inputs, outputs: _Model(input=inputs, output=outputs)

# from charidp.models.seq2seq.seq2seq_models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import logging

ln = logging.getLogger(__name__)


def get_model(
        story_maxlen, seq_maxlen, num_inputs, answer_maxlen, char_embedding_size=30,
        hidden_dim=150,
        encoder_depth=1,
        decoder_depth=1,
        conv_nfilters=[],
        query_lstm_dims=[],
        hierarchical_lstm_dims=[],
        query_maxlen=None,
        dropout=0.5,
        seq2seq_model="attention",
        hierarchical=False
):
    assert seq2seq_model in ["attention", "simple", "seq2seq"]

    if hierarchical:
        assert story_maxlen is not None, "Need story maxlen for hierarchical model"
        input_shape = (story_maxlen, seq_maxlen)

    else:
        input_shape = (seq_maxlen,)

    # TODO Need to manually add this for correct output dims? Probably not since we add our own dense layer on top
    # We WOULD need this if we eliminate last dense layer and have only an activation there.
    # hidden_dims_decoder.append(num_inputs)

    input1 = Input(shape=input_shape)
    inputs = [input1]

    embed_inner_input = Input((input_shape[-1],))
    embed1 = Embedding(num_inputs, char_embedding_size)(embed_inner_input)
    embed1 = Dropout(dropout)(embed1)

    pool = embed1
    for conv_nfilter in conv_nfilters:
        conv_len3 = Convolution1D(conv_nfilter, 3, activation="tanh", border_mode="same")(pool)
        conv_len3 = Dropout(dropout)(conv_len3)
        conv_len2 = Convolution1D(conv_nfilter, 2, activation="tanh", border_mode="same")(pool)
        conv_len2 = Dropout(dropout)(conv_len2)
        pool1 = MaxPooling1D(pool_length=2, stride=2, border_mode='valid')(conv_len3)
        pool2 = MaxPooling1D(pool_length=2, stride=2, border_mode='valid')(conv_len2)

        # could try merge with sum here instead, and no dense layer
        pool = concatenate([pool1, pool2])
        pool = TimeDistributed(Dense(conv_nfilter, activation="tanh"))(pool)

    if hierarchical:
        ln.debug("Inner LSTM input len is %s" % (pool._keras_shape[1]))
        seq_embedding = pool
        for hierarchical_lstm_dim in hierarchical_lstm_dims[:-1]:
            seq_embedding = LSTM(hierarchical_lstm_dim, return_sequences=True)(seq_embedding)
        seq_embedding = LSTM(hierarchical_lstm_dims[-1], return_sequences=False)(seq_embedding)
        embed_inner_model = Model(embed_inner_input, seq_embedding)
        seqs_embedded = TimeDistributed(embed_inner_model)(input1)
    else:
        embed_inner_model = Model(embed_inner_input, pool)
        seqs_embedded = embed_inner_model(input1)

    ln.debug("Will attend over %s time steps" % seqs_embedded._keras_shape[1])

    if query_maxlen is not None:
        input2 = Input(shape=(query_maxlen,))
        inputs.append(input2)
        embed2 = Embedding(num_inputs, char_embedding_size)(input2)
        # conv_embed2 = Convolution1D(hidden_dim, 2, activation="relu", border_mode="same")(embed2)
        # pool2 = MaxPooling1D(pool_length=2, border_mode='valid')(conv_embed2)
        query_encoded = embed2
        for query_lstm_dim in query_lstm_dims[:-1]:
            query_encoded = LSTM(query_lstm_dim, return_sequences=True)(query_encoded)
        query_encoded = LSTM(query_lstm_dims[-1], return_sequences=False)(query_encoded)
        query_encoded = RepeatVector(seqs_embedded._keras_shape[1])(query_encoded)
        seqs_embedded = concatenate([seqs_embedded, query_encoded])

    if seq2seq_model == "attention":
        decoded = AttentionSeq2Seq(
            batch_input_shape=seqs_embedded._keras_shape,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            depth=(encoder_depth, decoder_depth),
            output_length=answer_maxlen,
            # TODO add dropout once it works
        )(seqs_embedded)
    elif seq2seq_model == "seq2seq":
        decoded = Seq2Seq(
            batch_input_shape=seqs_embedded._keras_shape,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            depth=(encoder_depth, decoder_depth),
            output_length=answer_maxlen,
            peek=True
        )(seqs_embedded)
    else:
        decoded = SimpleSeq2Seq(
            batch_input_shape=seqs_embedded._keras_shape,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            depth=(encoder_depth, decoder_depth),
            output_length=answer_maxlen,
            dropout=dropout
        )(seqs_embedded)

    pred = TimeDistributed(Dense(num_inputs, activation="softmax"))(decoded)
    model = Model(inputs=inputs, outputs=pred)
    return model