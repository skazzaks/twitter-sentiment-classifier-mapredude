# Author: Devon Fritz
# This file reads in a model and builds a classifier from it, which can than
# predict sentiments of unseen sentences
import argparse

class PerceptronClassifier():
    """
    Represents our perceptron classifier. This class takes in a model as its
    classification base. Calls to 'predict' return the predicted polarity of
    the given text
    """


    def __init__(self, model):
        self.model = model

    def predict(self, line):
        final_value = 0
        for word in line.split():
            if word in self.model:
                final_value += self.model[word]

        return 1 if final_value >= 0 else 0

# Extracts the model that we stored in a file and returns it
def extract_model_from_file(input_file):
    model = {}
    with open(input_file) as f:
        for line in f:
            word, value = line.split()
            word = word[1:-1]
            value = float(value)
            model[word] = value

    return model


def predict_all_values(classifier, test_file):
    with open(test_file) as f:
        total = 0
        total_right = 0

        for line in f:
            line = line.split(',')
            polarity = 1 if int(line[1]) == 1 else 0
            content = ','.join(line[3:])
            predicted_polarity = classifier.predict(content)

            if predicted_polarity == polarity:
                total_right += 1

            total += 1

        print "Total Correct: " + str(total_right) + "/" + str(total)
        print "Percentage Correct: " + str(float(total_right)/float(total))

# Main entry point into the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model")
    parser.add_argument("infile")
    args = parser.parse_args()
    model = args.model
    infile = args.infile

    model = extract_model_from_file(model)
    classifier = PerceptronClassifier(model)
    predict_all_values(classifier, infile)
