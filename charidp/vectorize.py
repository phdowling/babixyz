import logging
import numpy as np
from keras.preprocessing.sequence import pad_sequences

ln = logging.getLogger(__name__)


class Vectorizer(object):
    def __init__(
            self, story_maxlen=None, answer_maxlen=None, query_maxlen=None, questions=False, two_layered=False,
            sent_maxlen=None, sep_char=u"\n", delete_long=False
    ):
        self.story_maxlen = story_maxlen
        self.sent_maxlen = sent_maxlen
        self.answer_maxlen = answer_maxlen
        self.questions = questions
        self.query_maxlen = query_maxlen
        self.two_layered = two_layered
        self.sep_char = sep_char

        self.vocab_size = None

        self.char_indices = None
        self.indices_char = None

        self.delete_long = delete_long

        self.initialized = False

    def stringify_X(self, X, vectorized=True):
        if not self.two_layered:
            return self.stringify_Y(X, vectorized=vectorized)
        else:
            res = []
            if not vectorized:
                char_getter = lambda timestep: self.indices_char[timestep]
            else:
                char_getter = lambda timestep: self.indices_char[np.argmax(timestep)]

            for sents_sample in X:
                join_char = self.sep_char if self.sep_char is not None else " "
                pred_str = join_char.join(
                    [u"".join(
                        [char_getter(timestep).decode("utf-8") for timestep in sents]
                    ) for sents in sents_sample]
                )
                res.append(pred_str)
            return res

    def stringify_Y(self, Y, vectorized=True):
        res = []

        if not vectorized:
            char_getter = lambda timestep: self.indices_char[timestep]
        else:
            char_getter = lambda timestep: self.indices_char[np.argmax(timestep)]

        for sample in Y:
            pred_str = "".join([char_getter(timestep) for timestep in sample])
            res.append(pred_str)

        #else:
        #    for sample in Y:
        #        pred_str = u"".join([self.indices_char[np.argmax(timestep)] for timestep in sample])
        #        res.append(pred_str)
        return res

    def initialize(self, texts):
        ln.debug("Initializing char vectorizer!")

        all_chars = set()
        for tup in texts:
            for string in tup:
                all_chars |= set(string)
        char_vocab = sorted(list(all_chars))

        self.vocab_size = len(char_vocab) + 2  # one for input mask, one for output stop symbol
        self.char_indices = {c: i + 2 for i, c in enumerate(char_vocab)}
        self.char_indices["<END>"] = 1  # string end symbol
        self.indices_char = {i + 2: c for i, c in enumerate(char_vocab)}
        self.indices_char[0] = "."  # input mask
        self.indices_char[1] = "<END>"  # output end (not masked!)

        if self.two_layered:
            story_maxlen = max(map(lambda text: text.count(self.sep_char) + 1, (tup[0] for tup in texts)))
            if self.story_maxlen is None:
                self.story_maxlen = story_maxlen
            self.story_maxlen = min(story_maxlen, self.story_maxlen)

            sent_maxlen = max(
                map(
                    lambda text: max([len(sent) for sent in text.split(self.sep_char)]),
                    (tup[0] for tup in texts)
                )
            )
            if self.sent_maxlen is None:
                self.sent_maxlen = sent_maxlen
            self.sent_maxlen = min(sent_maxlen, self.sent_maxlen)
        else:
            actual_sent_maxlen = max(map(len, (tup[0] for tup in texts)))
            if self.sent_maxlen is None:
                self.sent_maxlen = actual_sent_maxlen
            self.sent_maxlen = min(self.sent_maxlen, actual_sent_maxlen)

        answer_maxlen = max(map(len, (tup[-1] for tup in texts))) + 1
        if self.answer_maxlen is None:
            self.answer_maxlen = answer_maxlen
        self.answer_maxlen = min(self.answer_maxlen, answer_maxlen)

        query_maxlen = max(map(len, (tup[1] for tup in texts)))
        if self.questions and self.query_maxlen is None:
            self.query_maxlen = query_maxlen
        self.query_maxlen = min(query_maxlen, self.query_maxlen)

        self.initialized = True

    def _vectorize_one(self, story, question=None):
        if self.two_layered:
            if len(story) > self.story_maxlen * self.sent_maxlen and self.delete_long:
                return
            story = story.split(self.sep_char)
        else:
            if len(story) > self.sent_maxlen and self.delete_long:
                return

        if self.two_layered:
            x = [[[self.char_indices[c] for c in sent] for sent in story]]
        else:
            x = [[self.char_indices[c] for c in story]]

        if question is not None:
            q_x = [self.char_indices[c] for c in question]
            x.append(q_x)

        return x

    def vectorize(self, *datasets):

        if not self.initialized:
            self.initialize(reduce(lambda l, r: l + r, datasets, []))

        res = []

        for dataset in datasets:  # (train, val, test):
            X = []
            Y = []

            for data_tuple in dataset:
                story, answer = data_tuple[0], data_tuple[-1]
                x = self._vectorize_one(story, None if not self.questions else data_tuple[1])

                if x is None:
                    continue

                X.append(x)
                Y.append(
                    [self.char_indices[c] for c in answer] +
                    [self.char_indices["<END>"]]
                )

            if self.questions:
                stories_seqs, questions_seqs = zip(*X)
            else:
                stories_seqs = [sample[0] for sample in X]

            # pad all sequences
            if self.two_layered:
                # pre-padding: do nothing if no input is given
                inner_padded = [pad_sequences(story_seq, maxlen=self.sent_maxlen) for story_seq in stories_seqs]
                outer_padded = [
                    np.vstack(
                        [
                            np.zeros((self.story_maxlen - len(seqs), self.sent_maxlen)),
                            seqs
                        ]
                    ) for seqs in inner_padded
                    ]
                dataset_inputs = [
                    np.array(outer_padded)
                ]
            else:
                dataset_inputs = [
                    # pre-padding: do nothing if no input is given
                    pad_sequences(stories_seqs, maxlen=self.sent_maxlen),
                ]

            if self.questions:
                dataset_inputs.append(pad_sequences(questions_seqs, maxlen=self.query_maxlen))

            dataset_outputs = np.array(
                [
                    np.eye(self.vocab_size)[y] for
                    y in
                    # post-padding: generate outputs, then do nothing after stop symbol
                    pad_sequences(Y, maxlen=self.answer_maxlen, padding="post", value=0)
                    ]
            )
            res.append([dataset_inputs, dataset_outputs])

        return res
