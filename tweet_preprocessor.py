# Author: Anwesh Tuladhar <anwesh.tuladhar@gmail.com>
# Website: https://anwesht.github.io/
__author__ = 'Anwesh Tuladhar'


from argparse import ArgumentParser
import json
import logging
from datetime import datetime
import time
import os
import itertools
from collections import defaultdict

from nltk.parse.corenlp import CoreNLPParser


LOG_FMT = '%(asctime)s: %(threadName)s: %(levelname)s: %(message)s'


def get_tweet_obj(tweet):
    """
    Retrieve the inner most tweet information.

    Quoted tweet gets preference over retweeted status.

    Args:
        tweet: The current level of tweet obj.

    Returns:
        tweet (json obj): The inner most tweet obj.
    """
    try:
        # Parse quoted status or fall through to next check
        if 'quoted_status' in tweet:
            return get_tweet_obj(tweet['quoted_status'])

        # Parse retweeted status or return as is.
        if 'retweeted_status' in tweet:
            return get_tweet_obj(tweet['retweeted_status'])
    except Exception as e:
        logging.error("Error while parsing tweet with id: {}\n{}".format(tweet['id'], e))
        return None

    return tweet


def process_tweet(tweet):
    """
    Process a tweet

    Args:
        tweet: The tweet obj from which to extract info.

    Returns:
        tweet_text, hashtags, urls : where,
                tweet_text (str): complete tweet text
                hashtags (list): list of hashtags in the tweet
                urls (list): list of urls in the tweet
    """
    tweet = get_tweet_obj(tweet)

    if tweet is None or tweet['lang'] != 'en':
        return '', [], []

    if tweet['truncated']:
        return parse_extended_tweet(tweet['extended_tweet'])
    else:
        return parse_tweet(tweet)


def parse_tweet(t):
    """
    Extract the complete tweet text, hashtags and urls from the given tweet object.
    Args:
        t (dict): The inner most tweet object.

    Returns:
        tweet_text, hashtags, urls : where,
                tweet_text (str): complete tweet text
                hashtags (list): list of hashtags in the tweet
                urls (list): list of urls in the tweet
    """
    try:
        tweet_text = t['text']
        hashtags, urls = parse_entities(t['entities'])
        return tweet_text, hashtags, urls
    except Exception as e:
        logging.warning("Failed to parse tweet obj: {}".format(json.dumps(t)))
        return '', [], []


def parse_extended_tweet(et):
    """
    Extract the complete tweet text, hashtags and urls from the given extended_tweet object.
    Args:
        et (dict): The inner most tweet object.

    Returns:
        tweet_text, hashtags, urls : where,
                tweet_text (str): complete tweet text
                hashtags (list): list of hashtags in the tweet
                urls (list): list of urls in the tweet
    """
    try:
        tweet_text = et['full_text']
        hashtags, urls = parse_entities(et['entities'])
        return tweet_text, hashtags, urls
    except Exception as e:
        logging.warning("Failed to parse extended tweet obj: {}".format(json.dumps(et)))
        return '', [], []


def parse_entities(e):
    """
    Parse the entities json object

    Args:
        e: the entities json object
            entities: {
                "hashtags" : [ ... ],
                "urls": [ ... ]
                "user_mentions": [ ... ]
            }

    Returns:
        tweet_text, hashtags, urls : where,
                tweet_text (str): complete tweet text
                hashtags (list): list of hashtags in the tweet
                urls (list): list of urls in the tweet
    """
    # Converting to default dict with type list so we don't have to worry about KeyError
    e = defaultdict(list, e)
    hashtags = []
    for h in e['hashtags']:
        hashtags.append(h['text'])

    urls = []
    for u in itertools.chain(e['urls'], e['media']):
        urls.append(u['url'])

    return hashtags, urls


def tokenize_tweet(t):
    """
    Use the Stanford's PTBTokenizer to tokenize the tweet.
    Requires Stanford CoreNLP Server to be running.
    For setup information see: [
        https://stanfordnlp.github.io/CoreNLP/index.html,
        https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/
    ]
    Here we use the wrapper from `nltk` package instead of stanfordcorenlp

    From the directory where you setup the stanford NLP, Run the server:
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000

    Args:
        t: The tweet to be tokenized

    Returns:
        t (generator): A generator for the list of tokens generated from the tweet.
    """
    parser = CoreNLPParser(url='http://localhost:9000/')

    return parser.tokenize(t)


def process_tweets(args):
    """
    Process the tweets in the given file.

    Extracts the complete tweet text and hashtags, tokenizes them using the Stanford Tokenizer and
    writes 1000 tweets per file. The tokenized tweet text will not include the hashtags and any urls.

    Output format:
        {"text": "<space separated tokens for tweet>", "hashtags": "<space separated hashtags>"}

    Args:
        args: The parsed arguments.

    Returns:
        void.
    """
    input_tweets_file = args.tweets_file
    if not os.path.isfile(input_tweets_file):
        logging.error("{} file does not exist.".format(input_tweets_file))
        raise FileNotFoundError()

    filename = input_tweets_file.split('/')[-1]

    with open(input_tweets_file, 'r') as itf:
        tt_start = time.time()

        successfully_parsed = 0
        total_tweets = 0
        num_chunk = 0
        finished = False

        while not finished:
            cur_tweets = 0
            # Create new file every `chunk` tweets.
            chunk_fname = os.path.join(args.out_directory, '%s_%04d.json' % (filename, num_chunk))

            with open(chunk_fname, 'a') as otf:
                while cur_tweets < args.chunk:
                    line = itf.readline()
                    if not line:
                        finished = True
                        break

                    cur_tweets += 1
                    total_tweets += 1
                    tweet = json.loads(line)

                    tweet_text, hashtags, urls = process_tweet(tweet)

                    if not tweet_text:
                        continue

                    for h in hashtags:
                        tweet_text = tweet_text.replace('#'+h, '')

                    for u in urls:
                        tweet_text = tweet_text.replace(u, '')

                    out_tweet = dict()

                    out_tweet['text'] = ' '.join(tokenize_tweet(tweet_text))
                    out_tweet['hashtags'] = ' '.join(hashtags)

                    otf.write(json.dumps(out_tweet) + '\n')

                    successfully_parsed += 1

            logging.info("Parsed {} out of {} successfully.".format(successfully_parsed, total_tweets))

            num_chunk += 1

            if args.max_tweets and successfully_parsed == args.max_tweets:
                break

        tt_end = time.time()

        logging.info("Total time taken for parsing {} out of {} tweets: {} seconds.".format(successfully_parsed,
                                                                                            total_tweets,
                                                                                            tt_end-tt_start))


# -----------------------
# Setup script arguments
# -----------------------
parser = ArgumentParser(description=__doc__)
parser.add_argument('tweets_file',
                    help='Tweets JSON file to be processed.')
parser.add_argument('-c', '--chunk',
                    type=int,
                    default=5000,
                    help='Number of tweets to write per file.')
parser.add_argument('-od', '--out-directory',
                    default='./tokenized/',
                    help='Output directory for processed tweets.')
parser.add_argument('-m', '--max-tweets',
                    help='Maximum number of tweets to process.')
parser.add_argument('-D', '--debug', action='store_const',
                    help='Print debug messages', dest='logging_level',
                    default=logging.INFO, const=logging.DEBUG)


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FMT, level=args.logging_level,
                        filename="output.{}.log".format(datetime.now()), filemode='a')
    logging.info("Starting preprocessing tweets with args: {}".format(args))

    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    process_tweets(args)
