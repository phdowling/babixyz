# coding=utf-8
import logging

from charidp.load_data import load_namecopy_data
from charidp.tasks.modelcontrol.charmodelcontroller import CharModelController


def run_namecopy():
    samples_train, samples_test = load_namecopy_data(
        num_train=15000, num_test=1000
    )

    # model = "custom_decoder"
    model = "seq2seq_simple"
    # model = "seq2seq"
    # model = "attention"

    nl = CharModelController(use_model=model, config=config)
    nl.train(samples_train, samples_test, )
