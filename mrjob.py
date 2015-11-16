from mrjob.job import MRJob
from mrjob.step import MRStep
import re

# An implementation of the MRJob class, which defines methods that allow for the
# interaction with Map/Reduce
class ClassifyText(MRJob):

    def mapper_init():
        # Initialize the word feature vector that should be shared among all
        # instances of the mapper
        word_vector = []


    # The mapper takes in a line from the dataset and creates a feature vector
    # based on a bag of words approach. I Ita
    def mapper(self, _, line):
        # TODO DSF - this method should take a line and build the vector of bag
        # of words
        words = []
        for word in line:
            word_found = False
            for p in words:
                if word == p[0]:
                    p[1] += 1
                    word_found = True
            if not word_found:
                words.append([word, 1])

        # Combine this array with the current array that I have
        word_vector.append(words)
        # What do I return here?
        return 1, words

    def mapper_final():


    def reducer():

