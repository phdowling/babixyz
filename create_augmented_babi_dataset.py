from keras.utils.data_utils import get_file
from charidp.load_data import get_additional_names, split_names, babi_tasks, known_names
import random
import tarfile
import logging
import os
import unicodedata
import sys

if __name__ == "__main__":
    random.seed(31337)

alphabet = "abcdefghijklmnopqrstuvwxyz"
vowels = "aeiouy"
consonants = "bcdfghjklmnpqrstvwxz"
location_names = set()
known_locations = ["bathroom", "bedroom", "garden", "kitchen", "office", "hallway", "school", "park", "cinema"]

while len(location_names) < 10000:
    word = ""
    word += random.choice(consonants)  # .upper()
    for i in range(random.choice([1, 1, 2, 2, 2, 3])):
        word += "".join([random.choice(vowels) for _ in range(random.choice([1, 1, 1, 1, 2]))])
        word += random.choice(consonants)
    word += "".join([random.choice(vowels) for _ in range(random.choice([1, 1, 1, 1, 2]))])
    if word not in location_names and word not in ("many", "color"):
        location_names.add(word)

location_names = list(location_names)

tbl = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P'))


def remove_punctuation(text):
    return text.translate(tbl)

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-18s: %(message)s", level=logging.DEBUG)
ln = logging.getLogger(__name__)


def augment_babi_data(
        additional_stories, outdir, title="qa2", ten_k=True, val_split=0.1, names_test_split=0.1
):
    try:
        path = get_file('babi-tasks-v1-2.tar.gz',
                        origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
    except:
        print('Error downloading dataset, please download it manually:\n'
              '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
              '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
        raise
    tar = tarfile.open(path)
    challenge = 'tasks_1-20_v1-2/en{ten_k}/{title}_{set}.txt'
    out_challenge = os.path.join(outdir, 'tasks_1-20_v1-2_augmented/en{ten_k}/{title}_{set}.txt')

    outdir = os.path.join(outdir, 'tasks_1-20_v1-2_augmented/en{ten_k}/'.format(ten_k=("-10k" if ten_k else "")))
    if not os.path.exists(outdir):
        ln.info("Making path %s" % outdir)
        os.makedirs(outdir)

    ten_k = "-10k" if ten_k else ""

    ln.info("Replacing names randomly!")
    names = get_additional_names(caps=True)

    ln.debug("Ensuring no name overlap between train and test set.")
    random.shuffle(names)
    names_train, names_valid, names_test = split_names(
        names, 1.0 - names_test_split - val_split, names_test_split
    )
    lnames_train, lnames_valid, lnames_test = split_names(
        location_names, 1.0 - names_test_split - val_split, names_test_split
    )

    ln.debug("%s names in train, %s in val, %s in test" % (
        len(names_train), len(names_valid), len(names_test)
    ))

    ln.debug("%s location names in train, %s in val, %s in test" % (
        len(lnames_train), len(lnames_valid), len(lnames_test)
    ))

    augment_stories(
        out_challenge.format(title=babi_tasks[title], set='train', ten_k=ten_k),
        tar.extractfile(
            challenge.format(title=babi_tasks[title], set='train', ten_k=ten_k)
        ),  replace_names=names_train, replace_locations=lnames_train, from_split=val_split,
        additional_stories=additional_stories
    )

    augment_stories(
        out_challenge.format(title=babi_tasks[title], set='val', ten_k=ten_k),
        tar.extractfile(
            challenge.format(title=babi_tasks[title], set='train', ten_k=ten_k)
        ), replace_names=names_valid, replace_locations=lnames_valid, until_split=val_split)

    augment_stories(
        out_challenge.format(title=babi_tasks[title], set='test', ten_k=ten_k),
        tar.extractfile(
            challenge.format(title=babi_tasks[title], set='test', ten_k=ten_k)
        ), replace_names=names_test, replace_locations=lnames_test,
    )


def augment_stories(
        outfname, f, replace_names, replace_locations,
        from_split=None, until_split=None, additional_stories=0
):
    with open(outfname, "w") as outf:
        all_stories = []
        story = []

        all_lines = list(f.readlines())

        for idx, line in enumerate(all_lines):
            line = line.decode('utf-8').strip()
            nid, _ = line.split(' ', 1)

            if nid == "1" and idx != 0:
                story_full = "\n".join(story)
                story_full = augment_one_story(replace_names, replace_locations, story_full)
                all_stories.append(story_full)
                story = []

            story.append(line)

        story_full = "\n".join(story)
        story_full = augment_one_story(replace_names, replace_locations, story_full)
        all_stories.append(story_full)

        num_stories = len(all_stories)
        from_idx = 0 if from_split is None else int(from_split * num_stories)
        to_idx = num_stories if until_split is None else int(until_split * num_stories)
        stories_keep = all_stories[from_idx: to_idx]

        ln.info("Augmented %s stories.." % len(stories_keep))
        for story in stories_keep:
            outf.write(story + "\n")

        if additional_stories != 0:
            ln.info("Generating %s additional stories.." % additional_stories)
            for _ in range(additional_stories):
                story_full = random.choice(stories_keep)
                new_story = augment_one_story(replace_names, replace_locations, story_full)
                outf.write(new_story + "\n")


def augment_one_story(replace_names, replace_locations, story_full):
    tokens = remove_punctuation(story_full).split()

    # all_names = set(map(lambda s: s.lower(), known_names + replace_names))
    # names_in_story = set([token for token in tokens if token.lower() in all_names])
    names_in_story = set([token for token in tokens if token in known_names])
    used_names = set(list(names_in_story)[:] + [""])
    for name in names_in_story:
        replacement = ""
        while replacement.lower() in used_names:
            replacement = random.choice(replace_names)
        story_full = story_full.replace(name, replacement)
        used_names.add(replacement)

    # all_locations = set(map(lambda s: s.lower(), known_locations + replace_locations))
    # locations_in_story = set([token for token in tokens if token.lower() in all_locations])
    locations_in_story = set([token for token in tokens if token in known_locations])
    used_locations = set(list(names_in_story)[:] + [""])
    for lname in locations_in_story:
        replacement = ""
        while replacement.lower() in used_locations:
            replacement = random.choice(replace_locations)
        story_full = story_full.replace(lname, replacement)
        used_locations.add(replacement)

    return story_full


if __name__ == "__main__":
    for ten_k in [False, True]:
        for i in range(1, 21):
            print "10k %s, i=%s" % (ten_k, i)
            augment_babi_data(
                4000 if ten_k else 400, "data", title="qa%s" % i, ten_k=ten_k, val_split=0.1, names_test_split=0.1
            )

