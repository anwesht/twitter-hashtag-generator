# Twitter hashtag generator
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f2e8118b369c436d9f4dc91fba7d7c21)](https://app.codacy.com/app/anwesht/twitter-hashtag-generator?utm_source=github.com&utm_medium=referral&utm_content=anwesht/twitter-hashtag-generator&utm_campaign=Badge_Grade_Dashboard)
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Final project for CIS 6930 Social Media Mining. 

Using text summarization technique using recurrent neural networks for hashtag generation.

## Stanford Core NLP
This project uses Stanford's PTBTokenizer to tokenize the tweet.
Requires Stanford CoreNLP Server to be running.
For setup information see: 
    - [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html)
    - [Stanford CoreNLP Python](https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/)

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

## Using hashtag_generator_keras.py
This script can be used to train the RNN model and to evaluate the trained model.
The default arguments are setup to train the model. To load and run the inference loop, set the --load-model flag. The
model to be loaded is defined by params --load-model-path and --model-num.

```commandline
usage: hashtag_generator_keras.py [-h] [-dp DATA_PATH] [-tb TWEET_VOCAB]
                                  [-htv HASHTAG_VOCAB] [-m MODE] [-l LOG_ROOT]
                                  [-mr MODEL_ROOT] [-e EPOCH]
                                  [-minvc MIN_VOCAB_COUNT] [-hd HIDDEN_DIM]
                                  [-ed EMB_DIM] [-bs BATCH_SIZE]
                                  [-mes MAX_ENC_STEPS] [-maxds MAX_DEC_STEPS]
                                  [-minds MIN_DEC_STEPS] [-lr LEARNING_RATE]
                                  [-mrp LOAD_MODEL_PATH] [-lm] [-mn MODEL_NUM]
                                  [-D]

Script to train and use the hashtag generator RNN model

optional arguments:
  -h, --help            show this help message and exit
  -dp DATA_PATH, --data-path DATA_PATH
                        Path to tokenized json files.
  -tb TWEET_VOCAB, --tweet-vocab TWEET_VOCAB
                        Path expression to text vocabulary file.
  -htv HASHTAG_VOCAB, --hashtag-vocab HASHTAG_VOCAB
                        Path expression to hashtag vocabulary file.
  -m MODE, --mode MODE  must be one of train/eval/decode
  -l LOG_ROOT, --log-root LOG_ROOT
                        Root directory for all logging.
  -mr MODEL_ROOT, --model-root MODEL_ROOT
                        Root directory for all models.
  -e EPOCH, --epoch EPOCH
                        dimension of RNN hidden states
  -minvc MIN_VOCAB_COUNT, --min-vocab-count MIN_VOCAB_COUNT
                        Minimum word count to include in vocab
  -hd HIDDEN_DIM, --hidden-dim HIDDEN_DIM
                        dimension of RNN hidden states
  -ed EMB_DIM, --emb-dim EMB_DIM
                        dimension of word embeddings
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        minibatch size
  -mes MAX_ENC_STEPS, --max-enc-steps MAX_ENC_STEPS
                        max timesteps of encoder (max source text tokens)
  -maxds MAX_DEC_STEPS, --max-dec-steps MAX_DEC_STEPS
                        max timesteps of decoder (max summary tokens)
  -minds MIN_DEC_STEPS, --min-dec-steps MIN_DEC_STEPS
                        Minimum sequence length of generated summary. Applies
                        only for beam search decoding mode
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate
  -mrp LOAD_MODEL_PATH, --load-model-path LOAD_MODEL_PATH
                        Path to saved models to load.
  -lm, --load-model     Flag to load trained model in --load-model-path,
                        --model-num and run the inference loop.
  -mn MODEL_NUM, --model-num MODEL_NUM
                        The model number to load. This is based on the format
                        of the saved models during training.
  -D, --debug           Print debug messages
```


For documentation on the code see: [Wiki](https://github.com/anwesht/twitter-hashtag-generator/wiki/Documentation)
