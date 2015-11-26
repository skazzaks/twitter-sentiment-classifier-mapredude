from mrjob.job import MRJob


class TwitterTextClassifierMRJob(MRJob):
    # An implementation of the MRJob class, which defines methods that allow for
    # the interaction with Map/Reduce

    def perform_update(self, learning_rate, training_vector, polarity, value):
        for v in training_vector:
            # update the weight
            self.word_weights[v] = self.word_weights[v] + \
                (learning_rate * ((polarity) - value))

    def mapper_init(self):
        self.word_map = {}
        self.word_weights = []
        self.dont = -1

    def mapper(self, _, line):
        LEARNING_RATE = .1
        # First, break up the line into its polarity and content
        line = line.split(',')
        polarity = 1 if int(line[1]) == 1 else -1
        content = ','.join(line[3:])

        # The words from the line of text we got
        words = unicode(content, 'utf-8').split()

        # Go through the words from this line and add them two our 2 persistent
        # structures, capturing their indices for quick access later
        feature_list = []
        for w in words:
            index = -1
            if w not in self.word_map.keys():
                index = len(self.word_map.keys())
                self.word_map[w] = index
                self.word_weights.append(0)
            else:
                index = self.word_map[w]


            if w == "don't" and self.dont == -1:
                self.dont = index

            feature_list.append(index)

        # Now that we have the features for this record, let's run the
        # pereceptron algorithm to update our weighted vector
        value = 0
        for v in feature_list:
            value += self.word_weights[v]

        # we have misclassified if
        # 1) the example had a positive polarity and we didn't get a
        # positive number or
        # 2) the example had a negative polarity and we didn't get a
        # negative number
        if ((polarity * value) <= 0):
            self.perform_update(LEARNING_RATE, feature_list, polarity, value)

    def mapper_final(self):
        # Now that we filled up the vector with values, let's output them for
        # the reducer
        for k, v in self.word_map.iteritems():
            yield (k, self.word_weights[v])

    def reducer(self, word, weights):
        total = 0
        elements = 0
        for w in weights:
            total += w
            elements += 1

        total = total / elements

        yield (word, total)

if __name__ == "__main__":
    TwitterTextClassifierMRJob.run()
