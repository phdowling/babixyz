# coding=utf-8
import os
import tarfile
import zipfile
import random
from keras.utils.data_utils import get_file
import logging
ln = logging.getLogger(__name__)

known_names_cap = [
    "Daniel", "John", "Mary", "Sandra", "Fred", "Bill", "Yann", "Julie", "Jessica", "Emily", "Winona", "Gertrude",
    "Brian", "Julius", "Greg", "Bernhard", "Lily", "Jason", "Antoine", "Jeff", "Sumit"
]
known_names_lower = [n.lower() for n in known_names_cap]
known_names = known_names_cap + known_names_lower

babi_tasks = {
    "qa1": "qa1_single-supporting-fact",
    "qa2": "qa2_two-supporting-facts",
    "qa3": "qa3_three-supporting-facts",
    "qa4": "qa4_two-arg-relations",
    "qa5": "qa5_three-arg-relations",
    "qa6": "qa6_yes-no-questions",
    "qa7": "qa7_counting",
    "qa8": "qa8_lists-sets",
    "qa9": "qa9_simple-negation",
    "qa10": "qa10_indefinite-knowledge",
    "qa11": "qa11_basic-coreference",
    "qa12": "qa12_conjunction",
    "qa13": "qa13_compound-coreference",
    "qa14": "qa14_time-reasoning",
    "qa15": "qa15_basic-deduction",
    "qa16": "qa16_basic-induction",
    "qa17": "qa17_positional-reasoning",
    "qa18": "qa18_size-reasoning",
    "qa19": "qa19_path-finding",
    "qa20": "qa20_agents-motivations",
}


def get_additional_names(caps=False):
    try:
        path = get_file('census-derived-all-first.txt',
                        origin="http://deron.meranda.us/data/census-derived-all-first.txt")
    except:
        print('Error downloading dataset, please download it manually:\n'
              '$ wget http://deron.meranda.us/data/census-derived-all-first.txt\n'
              '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/census-derived-all-first.txt')
        raise

    names = []
    for line in open(path, "r"):
        if line.strip():
            name = line.split()[0]
            name = name[0] + name[1:].lower()
            name = name if caps else name.lower()
            names.append(name)

    ln.debug("Loaded %s additional names" % len(names))
    return names


def split_names(names, split_1, split_2):
    n = len(names)
    till_1 = int(split_1 * n)
    till_2 = int(split_2 * n)
    names_train = names[:till_1]
    names_valid = names[till_1: -till_2]
    names_test = names[-till_2:]
    return names_train, names_valid, names_test


def get_babixyz_data(
        only_supporting=False, caps=False, title="qa2", ten_k=True
):
    try:
        path = get_file('babi-tasks-v1-2_augmented.tar.gz',
                        origin='http://philipp.dowling.io/data/tasks_1-20_v1-2_augmented.tar.gz')
    except:
        print('Error downloading dataset, please download it manually:\n'
              '$ wget http://philipp.dowling.io/data/tasks_1-20_v1-2_augmented.tar.gz\n'
              '$ mv tasks_1-20_v1-2_augmented.tar.gz ~/.keras/datasets/babi-tasks-v1-2_augmented.tar.gz')
        raise

    tar = tarfile.open(path)

    challenge = 'tasks_1-20_v1-2_augmented/en{ten_k}/{title}_{set}.txt'
    ten_k = "-10k" if ten_k else ""

    train = prep_stories(
        tar.extractfile(
            challenge.format(title=babi_tasks[title], set='train', ten_k=ten_k)
        ), caps=caps, only_supporting=only_supporting
    )

    val = prep_stories(
        tar.extractfile(
            challenge.format(title=babi_tasks[title], set='val', ten_k=ten_k)
        ), caps=caps, only_supporting=only_supporting)

    test = prep_stories(
        tar.extractfile(
            challenge.format(title=babi_tasks[title], set='test', ten_k=ten_k)
        ), caps=caps, only_supporting=only_supporting
    )

    return train, val, test


def clean_text(text, caps=False, additional=None):
    if not caps:
        text = text.lower()

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet += alphabet.upper() if caps else ""
    additional = "" if additional is None else additional

    text = filter(lambda c: c in additional + alphabet + "#\t\n\"()[].?!',:;- 0123456789", text)

    return text


def remove_punctuation(word):
    return word.replace(".", "").replace("?", "").replace(",", "")


def prep_stories(
        f, caps=False, only_supporting=False, max_length=None
):
    # Taken partially from keras babi sample code, adapted to char level code
    all_data = []
    story = []

    all_lines = list(f.readlines())

    for line in all_lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = clean_text(q, caps=caps)
            a = clean_text(a, caps=caps)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            all_data.append((substory, q, a))
            story.append('')
        else:
            sent = clean_text(line, caps=caps)
            story.append(sent)

    flatten = lambda data_: reduce(lambda _x, _y: _x + "\n" + _y, data_)
    all_data = [(flatten(story), q, answer) for story, q, answer in all_data if
                not max_length or len(flatten(story)) < max_length]

    random.shuffle(all_data)

    return all_data


def load_namecopy_data(num_train=80000, num_test=100, rand_name_range=[4, 7]):
    alphabet = u"abcdefghijklmnopqrstuvwxyzöäü"
    alphabet_upper = alphabet#.upper()
    punct = u"1234567890!?,.-/()="
    chars = alphabet + alphabet_upper + punct

    def random_letter_sequence(length, nameish=False):
        if nameish:
            return random.choice(alphabet_upper) + "".join([random.choice(alphabet) for _ in range(length - 1)])
        else:
            return "".join([random.choice(chars) for _ in range(length)])

    def generate_samples(names, num_samples, minlen=20, maxlen=40):
        samples = []
        for _ in range(num_samples):
            sample_len = random.randint(minlen, maxlen)
            name = random.choice(names)
            offset = len(name) + len("<n></n>")
            start_idx = random.randint(0, sample_len - offset)
            sample = (
                random_letter_sequence(start_idx) +
                "<n>" + name + "</n>" +
                random_letter_sequence(sample_len - (start_idx + offset))
            )
            samples.append((sample, name))

        return samples

    names_train = [random_letter_sequence(random.randint(*rand_name_range), nameish=True) for _ in range(num_train)]
    names_test = [
        u"John", u"Bill", u"Marie", u"Carl", u"Steve", u"Kate", u"Peter", u"Ron", u"Harvey", u"Arthur", u"Tom",
        u"Howard", u"James", u"Tony", u"Ricky", u"Donald"
    ]

    samples_train = generate_samples(names_train, num_train)
    samples_test = generate_samples(names_train + names_test, num_test)

    return samples_train, samples_test


def load_names_cities_data(num_train=1000, val_split=0.1, num_test=1000, caps=True):
    num_valid = int(num_train * val_split)

    def generate_samples(names, cities, num_samples):
        samples = []
        for _ in range(num_samples):
            n = random.choice(names)
            name = "<n>" + n + "</n>"
            city = "<l>" + random.choice(cities) + "</l>"
            sample = [name, city]  # TODO maybe add some noise
            random.shuffle(sample)
            sample = " ".join(sample)
            samples.append((sample, n))

        return samples

    all_names = get_additional_names(caps=caps)
    random.shuffle(all_names)
    names_train, names_valid, names_test = split_names(all_names, 0.9 - val_split, 0.1)

    ln.debug("%s names in train, %s in val, %s in test" % (
        len(names_train), len(names_valid), len(names_test)
    ))

    # names_train = all_names[:int(4. * (len(all_names) / 5.))]
    # n_n_train = len(names_train)
    # split_idx = int(n_n_train * val_split)
    # names_train, names_valid = names_train[split_idx:], names_train[:split_idx]
    # names_test = all_names[int(len(all_names) / 5.):]

    # ln.debug("%s names in train, %s in val, %s in test" % (
    #     len(names_train), len(names_valid), len(names_test)
    # ))

    # names_train = all_names[:(len(all_names) / 2)]
    # names_test = all_names[(len(all_names) / 2):]

    all_cities = load_city_names(caps=caps)
    cnames_train, cnames_valid, cnames_test = split_names(all_cities, 0.9 - val_split, 0.1)
    # random.shuffle(all_cities)
    # cnames_train = all_cities[:int(4. * (len(all_cities) / 5.))]
    # cn_n_train = len(cnames_train)
    # csplit_idx = int(cn_n_train * val_split)
    # cnames_train, cnames_valid = cnames_train[csplit_idx:], cnames_train[:csplit_idx]
    # cnames_test = all_cities[int(len(all_cities) / 5.):]

    ln.debug("%s city names in train, %s in val, %s in test" % (
        len(cnames_train), len(cnames_valid), len(cnames_test)
    ))

    # cnames_train = all_cities[:(len(all_cities) / 2)]
    # cnames_test = all_cities[(len(all_cities) / 2):]

    samples_train = generate_samples(names_train, cnames_train, num_train)
    samples_valid = generate_samples(names_valid, cnames_valid, num_valid)
    samples_test = generate_samples(names_test, cnames_test, num_test)

    return samples_train, samples_valid, samples_test


def load_city_names(country="GB", single_word=True, caps=True, city_name_maxlen=15):
    try:
        path = get_file("worldcities.zip", origin="http://www.opengeocode.org/download/worldcities.zip")
    except:
        print('Error downloading citites dataset, please download it manually:\n'
              '$ wget http://www.opengeocode.org/download/worldcities.zip\n'
              '$ mv worldcitites.zip ~/.keras/worldcities.zip')
        raise
    zipf = zipfile.ZipFile(path, "r")
    city_names = []
    for line in zipf.open("worldcities.csv"):
        line = line.decode("utf-8")
        if not caps:
            line = line.lower()

        try:
            country_code, _, _, _, _, lang, cityname, _, _ = line.strip().split(",")
        except:
            continue

        if country is not None and country_code.lower() != country.lower():
            continue
        cityname = cityname.split()
        if single_word and len(cityname) != 1:
            continue
        cityname = cityname[0][1:-1]
        if len(cityname) >= city_name_maxlen:
            continue

        city_names.append(cityname)

    zipf.close()

    return city_names
