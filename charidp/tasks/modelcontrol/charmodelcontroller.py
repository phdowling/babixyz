import new
import time
import random
import logging
import os
import h5py
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop
from keras.utils.io_utils import ask_to_proceed_with_overwrite

from charidp.models.char_models import get_model

from charidp.util import shuffle_together

ln = logging.getLogger(__name__)


class CharModelController(object):
    def __init__(
            self, model_args, vectorizer, config, data_has_questions=False,
    ):
        self.questions = data_has_questions
        self.config = config
        self.vectorizer = vectorizer

        self.model = None
        self.model_args = model_args

    def build_model(self):
        self.model = get_model(**self.model_args)
        ln.info(
            "Using RMSprop with %s" % dict(lr=float(self.config["learning_rate"]), rho=0.9, epsilon=1e-08, decay=0.0)
        )
        optimizer = RMSprop(lr=float(self.config["learning_rate"]), rho=0.9, epsilon=1e-08, decay=0.0)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
        self.model.summary()

        # hot-patch saving and loading TODO remove once this is fixed in seq2seq
        def save_weights(_self, fname, overwrite=True):
            if not overwrite and os.path.isfile(fname):
                proceed = ask_to_proceed_with_overwrite(fname)
                if not proceed:
                    return
            outf = h5py.File(fname, 'w')
            weight = _self.get_weights()
            for i in range(len(weight)):
                outf.create_dataset('weight' + str(i), data=weight[i])
            outf.close()

        def load_weights(_self, fname):
            infile = h5py.File(fname, 'r')
            weight = []
            for i in range(len(infile.keys())):
                weight.append(infile['weight' + str(i)][:])
            _self.set_weights(weight)

        self.model.save_weights = new.instancemethod(save_weights, self.model, None)
        self.model.load_weights = new.instancemethod(load_weights, self.model, None)

    def train(
            self, train, test, max_iterations=2000, epochs_per_iter=1, val_split=0.1, val_data=None, minibatch_size=32
    ):

        if val_data:
            ln.debug("%s val samples were supplied" % len(val_data[1]))
            X_inputs, Y = train
            X_val_in, Y_val = val_data
        else:
            ln.debug("Splitting train set to get val dat")
            split_idx = int(train[1].shape[0] * (1. - val_split))
            X_inputs_all, Y_all = train

            X_inputs = [x_input[:split_idx] for x_input in X_inputs_all]
            X_val_in = [x_input[split_idx:] for x_input in X_inputs_all]

            Y, Y_val = Y_all[:split_idx], Y_all[split_idx:]

        tX_sents, tY = test

        save_fname = "charmodel%s.h5" % int(time.time())

        checkpoint = ModelCheckpoint(
            save_fname, monitor="val_acc", save_best_only=True, save_weights_only=True, verbose=1
        )

        lr_scheduler = ReduceLROnPlateau(monitor="val_acc", factor=0.15, patience=3, verbose=1, min_lr=0.00001)
        lr_scheduler.on_train_begin()
        lr_scheduler.on_train_begin = lambda logs: None

        early_stopping = EarlyStopping(monitor="val_acc", patience=6)
        early_stopping.on_train_begin()
        early_stopping.on_train_begin = lambda logs: None

        if self.model is None:
            self.build_model()

        try:
            for iteration in range(max_iterations):
                ln.info("Starting iteration %s/%s" % (iteration + 1, max_iterations))
                start = time.time()
                hist = self.model.fit(
                    X_inputs,
                    Y,
                    #batch_size=minibatch_size, epochs=epochs_per_iter,
                    minibatch_size, epochs_per_iter,
                    validation_data=(X_val_in, Y_val),
                    callbacks=[checkpoint, lr_scheduler, early_stopping],
                    verbose=2
                )

                if early_stopping.model.stop_training:
                    ln.debug("Early stopping triggered, stopping training.")
                    break

                X_sample_inputs = [X_input[:250] for X_input in X_inputs]
                val_sample_inputs = X_val_in

                ln.debug("Evaluating train and val set..")
                train_score = self.evaluate_zero_one(X_sample_inputs, Y[:250], debug_print=5)
                val_score = self.evaluate_zero_one(val_sample_inputs, Y_val, debug_print=8)

                ln.info("Current train set accuracy: %s" % train_score, )
                ln.info("Current validation accuracy: %s" % val_score)
                ln.debug("History: %s" % (hist.history,))
                ln.debug("Epoch took %s seconds" % (time.time() - start))

                # Shuffle data
                all_inputs = shuffle_together(X_inputs + [Y])
                X_inputs, Y = all_inputs[:-1], all_inputs[-1]

        except KeyboardInterrupt:
            ln.info("Interrupted, stopping training!")

        self.model.load_weights(save_fname)

        # ln.info("Trained for %s iterations, saving.." % iteration)
        loss, acc = self.model.evaluate(tX_sents, tY, batch_size=128)
        zero_one = self.evaluate_zero_one(tX_sents, tY, debug_print=10)

        ln.debug("Final test loss: %s. Eval acc: %s" % (loss, acc))
        ln.debug("Final test zero-one accuracy: %s" % zero_one)
        return loss, acc, zero_one

    def generate_prediction(self, X):
        preds = self.model.predict(X)
        res = np.eye(self.vectorizer.vocab_size)[preds.argmax(axis=2)]

        return res

    def evaluate_zero_one(self, X, Y, debug_print=0):
        pred = self.generate_prediction(X)
        if self.questions:
            strings = zip(
                self.vectorizer.stringify_X(X[0], vectorized=False),  # Stories may be multi-layered
                self.vectorizer.stringify_Y(X[1], vectorized=False),  # Questions look like Y
                self.vectorizer.stringify_Y(Y),
                self.vectorizer.stringify_Y(pred)
            )
            for story_str, q_str, target_str, pred_str in map(lambda _: random.choice(strings), range(debug_print)):
                ln.debug(
                    "Story: %s\n Q: %s\n -> Target: %s - Prediction: %s" % (
                        story_str, q_str, target_str, pred_str
                    )
                )
        else:
            if debug_print > 0:
                strings = zip(
                    self.vectorizer.stringify_X(X[0], vectorized=False),
                    self.vectorizer.stringify_Y(Y),
                    self.vectorizer.stringify_Y(pred)
                )
                for story_str, target_str, pred_str in map(lambda _: random.choice(strings), range(debug_print)):
                    ln.debug(
                        "Story: %s - Target: %s - Prediction: %s" % (
                            story_str, target_str, pred_str
                        )
                    )

        score = np.apply_along_axis(
            all, 1,  # all timesteps correct?
            np.apply_along_axis(all, 2, pred == Y)  # prediction of timestep correct?
        ).mean()  # average gives 0-1 loss of exact matches

        return score
