# Author: Devon Fritz
# This script takes in lists of sentences and preprocesses them so that the
# classifier can do a better job
import argparse
import csv
import re


def process_file(filename, handlers):
    """
    Takes in the file we want to process and does some preprocessing so that the
    classifier can perform better.
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            result = line[3]
            for h in handlers:
                result = h(result)

            print(','.join(line[0:3]) + ', ' + result)


def proc_multi_symbol(record):
    """
    Normalizes multiple symbols such as '!!!!!' to '!', since they express about
    the same thing and it will cut down on our word count.
    """
    # if they have as least two dots, capture it as an expression of an
    # ellipsis thought
    CONST_ELLIPSIS = ' ELL_NORM '
    record = re.sub(r'(\.\.+)', CONST_ELLIPSIS, record)

    # normalize the exclamation marks in a similar way
    CONST_EXCLAMATION = ' EXC_NORM '
    record = re.sub(r'(!+)', CONST_EXCLAMATION, record)
    return record


def proc_handle_links(record):
    """
    Removes the links that are commonly found in tweets and replaces them
    with a normalizing constant, so that the classifier can train more
    effectively
    """
    CONST_LINK = ' LINK_NORM '

    record = re.sub(r'http://[^\s]*', CONST_LINK, record)
    return record

def proc_lowercase(record):
    return record.lower()

def proc_handle_ats(record):
    """
    Removes the @someone links from Twitter's grammar and replaces them with a
    normalizing constant, so that the classifier can train more effectively
    """
    CONST_AT = ' LINK_AT '

    record = re.sub(r'@[^\s]*', CONST_AT, record)
    return record

# Main entry into the file - reads in our default file and preprocesses it, then
# spits out the output which can be piped into a postprocessed file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("infile")
    args = parser.parse_args()
    infile = args.infile

    process_file(infile, [proc_handle_ats,
                          proc_handle_links,
                          proc_multi_symbol,
                          proc_lowercase])
