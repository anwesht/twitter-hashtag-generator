# Twitter hashtag generator
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Final project for CIS 6930 Social Media Mining. 

Using text summarization technique using recurrent neural networks for hashtag generation.

## Stanford Core NLP
This project uses Stanford's PTBTokenizer to tokenize the tweet.
Requires Stanford CoreNLP Server to be running.
For setup information see: 
- https://stanfordnlp.github.io/CoreNLP/index.html
- https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/

Here we use the wrapper from `nltk` package instead of stanfordcorenlp

### Running the server
From the directory where you setup the stanford NLP, execute:
```commandline
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
```


## Using tweet_preprocessor.py
This script has two commands: 
1. *tokenize*: Parse and tokenize tweets.
2. *vocab*: Generate vocabulary files for the tokenized tweets.
```commandline
usage: tweet_preprocessor.py [-h] [-od OUT_DIRECTORY] [-D]
                             {tokenize,vocab} ...

Script to tokenize tweets and/or generate vocabulary from tokenized tweets.

optional arguments:
  -h, --help            show this help message and exit
  -od OUT_DIRECTORY, --out-directory OUT_DIRECTORY
                        Output directory for processed tweets.
  -D, --debug           Print debug messages

Sub-commands:
  Valid sub-commands are:

  {tokenize,vocab}      sub-command help
    tokenize            Parse and Tokenize the given tweets.
    vocab               Generate vocab files from tokenized tweet
                        files.Assumes tokenized tweets are in OUT_DIRECTORY
```

### Tokenizer
This command takes a file containing valid tweets as input and outputs multiple files 
each with `chunk` number of tokenized tweets.

The output file has json objects of format: 
{"text": "<space separated tokens for tweet>", "hashtags": "<space separated hashtags>"}

```commandline
usage: tweet_preprocessor.py tokenize [-h] [-c CHUNK] [-m MAX_TWEETS] [-g]
                                      tweets_file

positional arguments:
  tweets_file           Tweets JSON file to be processed.

optional arguments:
  -h, --help            show this help message and exit
  -c CHUNK, --chunk CHUNK
                        Number of tweets to write per file.
  -m MAX_TWEETS, --max-tweets MAX_TWEETS
                        Maximum number of tweets to process.
  -g, --gen-vocab       Generate vocab files while processing tweets.
```

### Vocab generator
This command reads all the tokenized json files in the output directory and generates two
vocab files: tweet_vocab.txt and hashtag_vocab.txt corresponding to the tokens present in
tweet text and hashtags respectively.

```commandline
usage: tweet_preprocessor.py vocab [-h]

optional arguments:
  -h, --help  show this help message and exit
```

For documentation on the code see: [Wiki](https://github.com/anwesht/twitter-hashtag-generator/wiki/Documentation)
