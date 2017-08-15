import logging
from charidp.tasks.babi_qa import run_babixyz_qa
from charidp.tasks.names_cities import run_names_cities
import json
import sys
import time

class StdToLog(object):
    def __init__(self, out):
        self.out = out

    def write(self, data):
        data = data.strip()
        if data:
            self.out(data)

    def flush(self, *args, **kwargs):
        pass

if __name__ == "__main__":
    import random
    logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)-18s: %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("babixyz.%s.out" % int(time.time()))
    consoleHandler = logging.StreamHandler()

    fileHandler.setFormatter(logFormatter)
    consoleHandler.setFormatter(logFormatter)

    rootLogger.addHandler(fileHandler)

    ln = logging.getLogger(__name__)
    seed = random.randint(0, 31337)
    random.seed(seed)
    ln.info("Seed is %s" % seed)

    logger = logging.getLogger("print_logger")
    sys.stdout = StdToLog(logger.debug)

    if len(sys.argv) != 2:
        ln.error("Please specify config file!")

    conffile = sys.argv[1].strip()

    with open(conffile, "r") as infile:
        config = json.load(infile)

    if config["TASK"] == "babixyz":
        ln.debug("Running bAbIxyz with %s" % config)
        run_babixyz_qa(config)

    elif config["TASK"] == "toy":
        ln.debug("Running toy with %s" % config)
        run_names_cities(config)

    ln.debug("Config was: %s" % config)
