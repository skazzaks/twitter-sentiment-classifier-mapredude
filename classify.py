# Author: Devon Fritz
# This file classifies the provided text based on its word features using
# SVM
import argparse
import csv
from sklearn import svm
import time
from scipy.sparse import csr_matrix
import numpy

def timeit(f, no_params=False):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print ("func:%r took: %2.4f sec" %
               (f.__name__, te-ts))
        return result

    return timed


class SVM_record:
    """
    This class represents one SVM record and stores internally its SVM
    representation.
    """
    def __init__(self, array, all_words):
        self.record_number = array[0]
        self.polarity = 1 if int(array[1]) == 1 else -1
        self.bag_of_words = self.calculate_bag_of_words(array[3], all_words)
        self.orig_sentence = array[3]

    def calculate_bag_of_words(self, message, all_words):
        message_array = message.split()
        bow = []

        for word in all_words:
            if word in message_array:
                bow.append(1)
            else:
                bow.append(0)

        return numpy.array(bow)


def create_word_array(infile):
    """Creates a huge word array that represents all words in the text"""
    all_words = []

    with open(infile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            all_words += line[3].split()

    return all_words


def process_file(infile, all_words, perc_test=20):
    """Collects all rows and builds up the bag of words for them so that they
    can feed the classifier"""
    with open(infile) as csvfile:
        reader = csv.reader(csvfile)
        records = []
        for line in reader:
            records.append(SVM_record(line, all_words))

    cutoff_record = int(len(records) * (perc_test/100))
    test_records = numpy.array(records[0:cutoff_record])
    training_records = numpy.array(records[cutoff_record:])
    return training_records, test_records

@timeit
def train_classifier(records):
    """This method takes the finished vector records we have build up and
    feeds them to an SVM classifier based on their polarity
    """
    vectors = []
    polarities = []

    for r in records:
        vectors.append(r.bag_of_words)
        polarities.append(r.polarity)

    print("Start training classifier")
    clf = svm.SVC(kernel='linear')
    clf.fit(csr_matrix(vectors), polarities)

    return clf


@timeit
def classify_unseen_records(classifier, records):
    right = 0
    total = 0
    total_1 = 0
    for r in records:
        result = classifier.predict(r.bag_of_words)
        if result[0] == r.polarity:
            right += 1

        if result[0] == 1:
            total_1 += 1
        total += 1

    print("total 1s: %i" % total_1)
    # print("expected: " + str(r.polarity) + " got: " + str(result[0]))

    print("total: %f (%i/%i)" % ((right/float(total)), right, total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("infile")
    args = parser.parse_args()
    infile = args.infile

    all_words = list(set(create_word_array(infile)))
    training_records, test_records = process_file(infile, all_words)
    classifier = train_classifier(training_records)
    results = classify_unseen_records(classifier, test_records)
