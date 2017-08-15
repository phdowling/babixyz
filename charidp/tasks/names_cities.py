# coding=utf-8
import logging

from charidp.load_data import load_names_cities_data
from charidp.tasks.modelcontrol.charmodelcontroller import CharModelController
from charidp.util import build_model_args
from charidp.vectorize import Vectorizer
ln = logging.getLogger(__name__)


def run_names_cities(config):
    caps = config["caps"]
    num_train = config["names_cities_samples"]
    samples_train, samples_valid, samples_test = load_names_cities_data(
        num_train=num_train,
        num_test=5000,
        caps=caps
    )

    ln.debug("Sample train: %s" % (samples_train[0],))
    ln.debug("Sample test: %s" % (samples_test[12],))

    vectorizer = Vectorizer(sent_maxlen=32, questions=False, sep_char="\n")

    train, val, test = vectorizer.vectorize(samples_train, samples_valid, samples_test)
    ln.debug("Num train samples: %s" % train[1].shape[0])

    model_args = build_model_args(config, vectorizer)

    nl = CharModelController(model_args, vectorizer, config=config, data_has_questions=False)
    nl.train(train, test, val_data=val, minibatch_size=config["minibatch_size"])
