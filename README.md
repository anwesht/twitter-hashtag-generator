# Twitter hashtag generator
Final project for CIS 6930 Social Media Mining. 

Using text summarization technique using recurrent neural networks for hashtag generation.

## Using tweet_preprocessor.py
This script has two commands: 
1. *tokenize*: Parse and tokenize tweets.
2. *vocab*: Generate vocabulary files for the tokenized tweets.
```
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

### Tokenizer:
This command takes a file containing valid tweets as input and outputs multiple files 
each with `chunk` number of tokenized tweets.

The output file has json objects of format: 
{"text": "<space separated tokens for tweet>", "hashtags": "<space separated hashtags>"}

```
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

### Vocab generator:
This command reads all the tokenized json files in the output directory and generates two
vocab files: tweet_vocab.txt and hashtag_vocab.txt corresponding to the tokens present in
tweet text and hashtags respectively.

```
usage: tweet_preprocessor.py vocab [-h]

optional arguments:
  -h, --help  show this help message and exit
```

For documentation on the code see: [Wiki](https://github.com/anwesht/twitter-hashtag-generator/wiki/Documentation)
