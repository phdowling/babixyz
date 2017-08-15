# coding=utf-8
import logging
from charidp.load_data import get_babixyz_data
from charidp.tasks.modelcontrol.charmodelcontroller import CharModelController
from charidp.util import build_model_args
from charidp.vectorize import Vectorizer
ln = logging.getLogger(__name__)


def run_babixyz_qa(config):
    hierarchical = config["hierarchical"]
    task = config["task_no"]
    ten_k = config["ten_k"]
    caps = config["caps"]

    samples_train, samples_valid, samples_test = get_babixyz_data(
        title=task,
        only_supporting=config["supporting_facts_only"], ten_k=ten_k, caps=caps,
    )

    ln.debug("Sample train: %s" % (samples_train[0],))
    ln.debug("Sample test: %s" % (samples_test[12],))

    if hierarchical:
        limit_stories = 10
        vectorizer = Vectorizer(story_maxlen=limit_stories, questions=True, two_layered=True, sep_char="\n")
    else:
        limit_stories = config["limit_stories_chars"]
        vectorizer = Vectorizer(sent_maxlen=limit_stories, questions=True, two_layered=False, delete_long=False)

    train, val, test = vectorizer.vectorize(samples_train, samples_valid, samples_test)
    ln.debug("Num train samples: %s" % train[1].shape[0])

    model_args = build_model_args(config, vectorizer, hierarchical=hierarchical)

    nl = CharModelController(model_args, vectorizer, config=config, data_has_questions=True)
    nl.train(train, test, minibatch_size=config["minibatch_size"], val_data=val)
