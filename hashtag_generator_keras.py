# Author: Anwesh Tuladhar <anwesh.tuladhar@gmail.com>
# Website: https://anwesht.github.io/
__author__ = 'Anwesh Tuladhar'

import numpy as np

import os
import time
from collections import namedtuple
import json

# from argparse import ArgumentParser
import logging
from datetime import datetime

from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dense
from keras import metrics, losses, optimizers
from keras.preprocessing import sequence
from keras.utils import to_categorical


# ---------
# Constants
# ---------
START_HASHTAG = '<h>'
END_HASHTAG = '</h>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'


class Vocab:
    def __init__(self, vocab_file, log_root, min_vocab_count=10):
        """
        Class to track vocabulary used in tweets and hashtags

        Args:
            vocab_file: Path to vocab file.

            Note: vocab_file is generated using tweet_preprocessor.py
            The format of this file is:
            <word> <number of times it appeared in the entire corpus>
        """
        self._wordToId = dict()
        self._idToWord = dict()

        self._id = 0
        # create id's for special tokens.
        # XXX: PAD_TOKEN has to have 0 index as embedding layer only supports mask_zero=True.
        for t in [PAD_TOKEN, UNKNOWN_TOKEN, START_HASHTAG, END_HASHTAG]:
            self._wordToId[t] = self._id
            self._idToWord[self._id] = t
            self._id += 1

        logging.info('Loading vocab file: {}'.format(vocab_file))

        skipped_vocabs = []
        with open(vocab_file, 'r') as vf:
            for line in vf:
                try:
                    w, cnt = line.split()
                except ValueError as e:
                    logging.warning('Skipping... Error unpacking vocab: \"{}\" in file: {}'.format(line.strip(),
                                                                                                   vocab_file))
                    continue

                if int(cnt) < min_vocab_count:
                    skipped_vocabs.append('{} {}'.format(w, cnt))
                    continue
                elif w.startswith('@'):
                    skipped_vocabs.append('{} {}'.format(w, cnt))
                    continue
                else:
                    e = str(w.encode('unicode-escape'))
                    if e[2] == '\\':
                        skipped_vocabs.append('{} {}'.format(w, cnt))
                        continue

                self._wordToId[w] = self._id
                self._idToWord[self._id] = w
                self._id += 1

        logging.info("Finished creating vocabulary object. Num words = {}".format(self._id))

        skipped_file = vocab_file.split('/')[-1] + '.skipped.txt'
        with open(os.path.join(log_root, skipped_file), 'w') as f:
            for s in skipped_vocabs:
                f.write(s + '\n')

        logging.info("Skipped {} words from file: {}. Skipped vocab stored in: {}".format(len(skipped_vocabs),
                                                                                          vocab_file,
                                                                                          skipped_file))

    def word2id(self, w):
        if w in self._wordToId:
            return self._wordToId[w]
        return self._wordToId[UNKNOWN_TOKEN]

    def id2word(self, i):
        if i in self._idToWord:
            return self._idToWord[i]

        logging.error("Id {} not in vocabulary".format(i))
        raise KeyError("Id {} not in vocabulary".format(i))

    def size(self):
        return self._id


class Example(object):
    """Class representing a train/val/test example."""

    def __init__(self, tweet, hashtags, tweet_vocab, hashtags_vocab, hps):
        """Initializes the Example. Produces the the encoder, decoder and target
        sequences, which are stored in self.

        Note: Out-Of-Vocabulary words -> OOV

        Args:
          tweet (str): source text: each token is separated by a single space.
          hashtags (str): Each hashtag is separated by a single space.
          tweet_vocab: Tweet Vocabulary object
          hashtags_vocab: Hashtag Vocabulary object
          hps: hyperparameters
        """
        self.hps = hps

        self.skip = False  # Flag to skip example if hashtag is none.

        # Get ids of special tokens
        start_decoding = hashtags_vocab.word2id(START_HASHTAG)
        stop_decoding = hashtags_vocab.word2id(END_HASHTAG)

        # Process the tweet
        tweet_words = tweet.split()  # list of strings
        self.enc_len = len(tweet_words)  # store the length of the tweet before padding
        self.enc_input = [tweet_vocab.word2id(w) for w in
                          tweet_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the hashtags
        hashtag_words = hashtags.split()  # list of strings
        hashtags_ids = [hashtags_vocab.word2id(w) for w in
                        hashtag_words]  # list of word ids; OOVs are represented by the id for UNK token

        hashtags_ids = [h for h in hashtags_ids if h != hashtags_vocab.word2id(UNKNOWN_TOKEN)]

        if len(hashtags_ids) == 0:
            # logging.warning('Skipping example as hashtag token is UNK.')
            self.skip = True
            return

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(hashtags_ids, start_decoding, stop_decoding)
        #  TODO: Using only 1 hashtag for now.
        # self.dec_input, self.target = self.get_dec_inp_targ_seqs(hashtags_ids[:(hps.max_dec_steps - 1)], start_decoding,
        #                                                          stop_decoding)
        self.dec_len = len(self.dec_input)

        # Store the original strings
        self.original_tweet = tweet
        self.original_hashtag = hashtags

    def get_dec_inp_targ_seqs(self, hashtags, start_id, stop_id):
        """Given the reference hashtags as a sequence of tokens, return the input sequence for the decoder,
        and the target sequence which we will use to calculate loss.
        The input sequence must start with the start_id and the target sequence must end with the stop_id.

        Args:
          hashtags: List of ids (integers)
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length starting with start_id
          target: sequence same length as input, ending with stop_id
        """
        inp = [start_id] + hashtags[:]  # pre-pend start token for inp
        target = hashtags[:]
        target.append(stop_id)  # append end token for target
        assert len(inp) == len(target)  # assert both have same length
        return inp, target


class Batch(namedtuple(typename="Batch",
                       field_names=("encoder_inputs", "decoder_inputs", "decoder_targets"))):
    """Wrapper class for named tuples, to allow for flexibily in returning different outputs."""
    pass


class Batcher:
    def __init__(self, filepath, tweet_vocab, hashtag_vocab, hps):
        self._filepath = filepath
        self._tweet_vocab = tweet_vocab
        self._hashtag_vocab = hashtag_vocab

        self._hps = hps

        self.create_training_batches()

    def get(self):
        if self._batch is not None:
            return self._batch
        else:
            logging.error('Batch has not been created.')
            raise Exception('Batch has not been created.')

    def create_training_batches(self):
        """
        Creates a training sequence for the current file.

        All sequences (encoder_inputs, decoder_inputs and decoder_targets) are padded to the max length of
        the corresponding sequence in this file.

        Required format of the input file:
        {'text': <tweet tokens>, 'hashtags': <hashtag tokens>}

        Shapes of sequences:
        encoder_inputs: [<num_examples>, <max_tweet_sequence_length>]
        decoder_inputs: [<num_examples>, <max_hashtag_sequence_length>]
        decoder_targets: [<num_examples>, <max_hashtag_sequence_length>, <num_hashtag_vocab>]

        The decoder targets is converted into a onehot encoded vector as the model ouputs categorical probabilities.
        """
        logging.info('Creating training batch for file: {}'.format(self._filepath))
        t1 = time.time()

        all_examples = list()
        skip_count = 0
        with open(self._filepath, 'r') as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                try:
                    tweet = obj['text']
                    hashtags = obj['hashtags']
                except KeyError as e:
                    logging.error('Given file({}) does not follow the required format'.format(self._filepath))
                    logging.error(e)
                    raise Exception(e)
                ex = Example(tweet, hashtags, self._tweet_vocab, self._hashtag_vocab, self._hps)
                if not ex.skip:
                    all_examples.append(ex)
                else:
                    skip_count += 1

        t2 = time.time()

        logging.info('Creating training batch: Finished reading file in {} seconds.'.format(t2 - t1))
        logging.info('Skipped {} examples as hashtag is UNK.'.format(skip_count))

        encoder_inputs = list()
        decoder_inputs = list()
        decoder_targets = list()

        for e in all_examples:
            encoder_inputs.append(e.enc_input)
            decoder_inputs.append(e.dec_input)
            decoder_targets.append(e.target)

        # One-hot encode the target sequence
        onehot_targets = to_categorical(sequence.pad_sequences(decoder_targets, padding='post'),
                                        num_classes=self._hashtag_vocab.size())

        # Create Batch
        self._batch = Batch(encoder_inputs=sequence.pad_sequences(encoder_inputs, padding='post'),
                            decoder_inputs=sequence.pad_sequences(decoder_inputs, padding='post'),
                            decoder_targets=onehot_targets)

        t3 = time.time()

        logging.info('Created training batch in: {} seconds. Total time taken: {}'.format(t3 - t2, t3 - t1))


class KerasModel:
    def __init__(self, hps, tweet_vocab, hashtag_vocab):
        """
        This class holds 3 models:
        1. Training Model: Encoder-Decoder RNN with:
            inputs: a. Encoder input: vectorized tweet tokens
                    b. Decoder input: vectorized hashtag token
            outputs: Decoder targets: onehot encoded hashtag tokens for training.

        2. Inference Encoder: Encoder for inference mode. Shares all weights with training model
        3. Inference Decoder: Decoder for inference mode. Shares all weights with training model

        Args:
            hps (NamedTuple): Hyper-parameters to use for the model
            tweet_vocab (Vocab): Instance of class Vocab for tweets
            hashtag_vocab (Vocab): Instance of class Vocab for hashtags
        """
        self.hps = hps
        self.tweet_vocab = tweet_vocab
        self.hashtag_vocab = hashtag_vocab

        if hps.load_model:
            self.load_model_from_file()
        else:
            self.build_graph()

    def load_model_from_file(self):
        logging.info('Loading trained model...')

        hps = self.hps
        info = "-{}.hdf5".format(hps.model_num)
        train_model = os.path.join(hps.load_model_path, 'train'+info)
        self._train_model = load_model(train_model)

        inf_enc_model = os.path.join(hps.load_model_path, 'inf_encoder' + info)
        self._inf_encoder_model = load_model(inf_enc_model)

        inf_dec_model = os.path.join(hps.load_model_path, 'inf_decoder' + info)
        self._inf_decoder_model = load_model(inf_dec_model)

    def build_graph(self):
        """ Build the Encoder-Decoder RNN network for training and inference. """
        logging.info('Building Encoder-Decoder RNN network')

        hps = self.hps

        # ---------------------------------
        # Create training Encoder RNN
        # ---------------------------------
        # Create input tensor to encoder RNN
        # shape = num_timesteps  => Timesteps will be the max of the batch.
        # Inputs will be padded to this length
        encoder_inputs = Input(shape=(None,),
                                     dtype='int32',
                                     name='encoder_inputs')

        # Encoder Embedding layer. Not giving input_length as it depends on each batch.
        # Add Masking. The inputs will be padded to the length of the max sequence in batch.
        encoder_embedding_layer = Embedding(input_dim=self.tweet_vocab.size(),
                                            output_dim=hps.emb_dim,
                                            mask_zero=True,
                                            name='enc_emb_layer')
        encoder_emb_inputs = encoder_embedding_layer(encoder_inputs)

        # Encoder LSTM layer
        encoder_layer = LSTM(units=hps.hidden_dim, return_state=True)
        encoder_outputs, enc_state_h, enc_state_c = encoder_layer(encoder_emb_inputs)
        encoder_states = [enc_state_h, enc_state_c]

        # ---------------------------------
        # Create training Decoder RNN
        # ---------------------------------
        # Create input tensor to decoder RNN
        # self._decoder_inputs = Input(shape=(hps.max_dec_steps,),
        decoder_inputs = Input(shape=(None,),
                                     dtype='int32',
                                     name='decoder_inputs')

        # adding a masking layer for decoder inputs.
        # dec_masked_inputs = Masking(mask_value=pad_id, input_shape=[None])(self._decoder_inputs)

        # Decoder Embedding layer. Not giving input_length as it depends on each batch
        decoder_embedding_layer = Embedding(input_dim=self.hashtag_vocab.size(),
                                            output_dim=hps.emb_dim,
                                            mask_zero=True,
                                            name='dec_emb_layer'
                                            )
        decoder_emb_inputs = decoder_embedding_layer(decoder_inputs)

        # Decoder LSTM layer
        decoder_layer = LSTM(units=hps.hidden_dim, return_state=True, return_sequences=True)
        decoder_outputs, unused_state_h, unused_state_c = decoder_layer(decoder_emb_inputs,
                                                                        initial_state=encoder_states)

        # ---------------------------------
        # Create Projection layer
        # ---------------------------------
        decoder_dense_layer = Dense(units=self.hashtag_vocab.size(), activation='softmax')
        dense_outputs = decoder_dense_layer(decoder_outputs)

        # ---------------------------------
        # Create Training Model
        # ---------------------------------
        # self._train_model = Model(inputs=[self._encoder_inputs, self._decoder_inputs],
        #                           outputs=dense_outputs)
        self._train_model = Model([encoder_inputs, decoder_inputs],
                                  dense_outputs)

        logging.info('Encoder-Decoder training Model Summary: \n')
        # self._train_model.summary()
        self._train_model.summary(print_fn=lambda x: logging.info(x + '\n'))  # write summary to file

        # ---------------------------------
        # Compile Training Model
        # ---------------------------------
        # Add Loss, Optimizer and Metrics
        optimizer = optimizers.Adam()  # TODO: Update parameters.
        self._train_model.compile(optimizer=optimizer,
                                  loss=losses.categorical_crossentropy,
                                  # loss=losses.sparse_categorical_crossentropy,
                                  metrics=[metrics.categorical_accuracy,
                                           losses.categorical_crossentropy])

        # ---------------------------------
        # Create Inference Encoder Model
        # ---------------------------------
        # Share the same encoder layer.
        self._inf_encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

        # ---------------------------------
        # Create Inference Decoder Model
        # ---------------------------------
        decoder_state_input_h = Input(shape=[hps.hidden_dim])
        decoder_state_input_c = Input(shape=[hps.hidden_dim])

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        inf_decoder_emb_inputs = decoder_embedding_layer(decoder_inputs)
        # Share the same decoder layer
        # inf_decoder_outputs, dec_state_h, dec_state_c = decoder_layer(decoder_emb_inputs,
        inf_decoder_outputs, dec_state_h, dec_state_c = decoder_layer(inf_decoder_emb_inputs,
                                                                      initial_state=decoder_states_inputs)
        inf_decoder_states = [dec_state_h, dec_state_c]

        # Inference dense layer
        inf_dense_outputs = decoder_dense_layer(inf_decoder_outputs)

        # Create the Inference Model
        self._inf_decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                                        [inf_dense_outputs] + inf_decoder_states)

        logging.info('Inference Decoder Model Summary: \n')
        # self._inf_decoder_model.summary()
        self._inf_decoder_model.summary(print_fn=lambda x: logging.info(x + '\n'))

    def run_training(self):
        """Runs training iterations, logging loss to screen and writing summaries"""
        hps = self.hps
        tokenized_files = [os.path.join(hps.data_path, p) for p in os.listdir(hps.data_path) if p.endswith('.json')]

        current_file_index = 0
        num_files = len(tokenized_files)
        # batcher = Batcher(tokenized_files[current_file_index], self.tweet_vocab, self.hashtag_vocab, self.hps)

        logging.info("Starting training loop...")

        total_epoch = 0

        while total_epoch < self.hps.epoch:
            # Get next batch. If current batcher is exhausted, create new batcher.
            # try:
            #     batch = batcher.get()
            # except Exception as e:
            if current_file_index == num_files:
                total_epoch += 1

            logging.info('Creating batcher for file: {}'.format(tokenized_files[current_file_index]))
            batcher = Batcher(tokenized_files[current_file_index], self.tweet_vocab, self.hashtag_vocab,
                              self.hps)
            batch = batcher.get()

            logging.info('Current batch is from file: {}'.format(batcher._filepath))

            logging.info('Running training step...')
            t0 = time.time()

            history = self._train_model.fit(x=[batch.encoder_inputs, batch.decoder_inputs],
                                            y=batch.decoder_targets,
                                            batch_size=self.hps.batch_size,
                                            validation_split=0.2)

            t1 = time.time()
            logging.info('Training step took: %.3f seconds.', t1 - t0)

            # Log the loss history for this training epoch
            logging.info('Training history: {}', history.history)

            self.save_model(current_file_index)

            if current_file_index % 5 == 0:
                self.predict_sequence(tokenized_files[(current_file_index + 1) % num_files])

            current_file_index = (current_file_index + 1) % num_files

    def save_model(self, epoch):
        # info = "-{epoch:02d}-{acc:.2f}.hdf5".format(epoch=epoch, acc=acc)
        info = "-{epoch:02d}.hdf5".format(epoch=epoch)
        self._train_model.save(filepath=os.path.join(self.hps.model_root, 'train'+info))
        self._inf_encoder_model.save(filepath=os.path.join(self.hps.model_root, 'inf_encoder'+info))
        self._inf_decoder_model.save(filepath=os.path.join(self.hps.model_root, 'inf_decoder'+info))

    def run_inference(self):
        hps = self.hps
        tokenized_files = [os.path.join(hps.data_path, p) for p in os.listdir(hps.data_path) if p.endswith('.json')]
        num_files = len(tokenized_files)

        logging.info("Starting inference loop...")
        total_epoch = 0
        total_accs = 0.0
        for current_file_index in range(num_files):
            if current_file_index == num_files:
                total_epoch += 1

            total_accs += self.predict_sequence(tokenized_files[current_file_index % num_files])

            # if current_file_index == 5:
            #     break

        logging.info('Average accuracy per chunk = {}/{} = {}'.format(total_accs, num_files, total_accs/num_files))

    # generate target given source sequence
    def predict_sequence(self, sourcefile, n_steps=5, num_tweets=10):
        batch = Batcher(sourcefile, self.tweet_vocab, self.hashtag_vocab, self.hps).get()

        logging.info('Predicting next batch...')
        all_outputs = list()

        tv = self.tweet_vocab
        hv = self.hashtag_vocab

        # for i, inp in enumerate(batch.encoder_inputs[:num_tweets]):
        for i, inp in enumerate(batch.encoder_inputs):
            # encode
            enc_state = self._inf_encoder_model.predict([inp])
            start_id = self.hashtag_vocab.word2id(START_HASHTAG)

            # start of sequence input
            # target_seq = np.array([0.0]).reshape(1, )
            # target_seq = np.zeros(shape=batch.decoder_inputs[i].shape)
            target_seq = np.zeros(shape=(1, 1))
            target_seq[0, 0] = start_id

            # collect predictions
            output = list()
            decoded_hashtag = ''
            for t in range(n_steps):
                # predict next hashtag
                yhat, h, c = self._inf_decoder_model.predict([target_seq] + enc_state)
                # store prediction
                output.append(yhat[0, 0, :])

                # ---------
                # Sample a token
                sampled_token_index = np.argmax(yhat[0, -1, :])
                sampled_char = self.hashtag_vocab.id2word(sampled_token_index)
                decoded_hashtag += ' ' + sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if sampled_char == END_HASHTAG:
                    break

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_token_index

                # update state
                enc_state = [h, c]

            target = np.array(output)
            all_outputs.append(target)

            tweet = [tv.id2word(t) for t in inp if t != tv.word2id(PAD_TOKEN)]
            actual = [hv.id2word(h) for h in batch.decoder_inputs[i] if h != hv.word2id(PAD_TOKEN)]

            for d in decoded_hashtag.split(' '):
                if d in actual:
                    logging.info("Decoded Hashtag: {}".format(decoded_hashtag))

                    logging.info('tweet={} actual={}, pred={}'.format(
                        ' '.join(tweet),
                        ' '.join(actual),
                        decoded_hashtag)
                    )

        logging.info('Calculating metrics on file:{}'.format(sourcefile))
        score = self._train_model.evaluate(x=[batch.encoder_inputs, batch.decoder_inputs],
                                           y=batch.decoder_targets)

        logging.info('Metrics: {}'.format(self._train_model.metrics_names))
        logging.info('Scores: {}'.format(score))

        acc = score[1]
        return acc

    # decode a one hot encoded string
    def one_hot_decode(self, encoded_seq):
        return [np.argmax(vector) for vector in encoded_seq]


def main(args):
    if not os.path.exists(args.log_root):
        if args.mode == 'train':
            os.makedirs(args.log_root)
        else:
            raise Exception("Logs directory {} doesn't exist. Run in train mode to begin.".format(args.log_root))

    logging.basicConfig(format=LOG_FMT, level=args.logging_level,
                        filename=os.path.join(args.log_root, "output.{}.log".format(datetime.now())), filemode='a')

    model_dir = os.path.join(args.model_root, str(datetime.now()))

    new_args = {}
    for k, v in args._asdict().items():
        if k == 'model_root':
            new_args[k]=model_dir
        else:
            new_args[k] = v

    args = MyArgs(**new_args)
    if not os.path.exists(model_dir):
        if args.mode == 'train':
            os.makedirs(model_dir)
        else:
            raise Exception("Models directory {} doesn't exist. Run in train mode to begin.".format(args.log_root))

    tweet_vocab = Vocab(args.tweet_vocab, args.log_root, args.min_vocab_count)
    hashtag_vocab = Vocab(args.hashtag_vocab, args.log_root)

    logging.info('Hyper-parameters: {}'.format(args))

    # Create Model
    model = KerasModel(args, tweet_vocab, hashtag_vocab)

    # Train Model OR Run Decoder.
    if not args.load_model:
        model.run_training()
    else:
        model.run_inference()


# ----------------------------------------------
# Setup script arguments
# These are common hyperparameter settings.
# ----------------------------------------------

LOG_FMT = '%(asctime)s: %(threadName)s: %(levelname)s: %(message)s'

"""
parser = ArgumentParser(description='Script to train and use the hashtag generator RNN model')

# Input data params
parser.add_argument('-dp', '--data-path',
                    type=str,
                    default='./tokenized_tweets4/',
                    help='Path to tokenized json files.')
parser.add_argument('-tb', '--tweet-vocab',
                    default='./tokenized_tweets4/tweets-4.json.tweet_vocab.txt',
                    help='Path expression to text vocabulary file.')
parser.add_argument('-htv', '--hashtag-vocab',
                    default='./tokenized_tweets4/tweets-4.json.hashtag_vocab.txt',
                    help='Path expression to hashtag vocabulary file.')

# Important settings
parser.add_argument('-m', '--mode',
                    type='str',
                    default='train',
                    help='must be one of train/eval/decode')

# Output params
parser.add_argument('-l', '--log-root',
                    type=str,
                    default='./logs',
                    help='Root directory for all logging.')

parser.add_argument('-mr', '--model-root',
                    default='./models',
                    help='Root directory for all models.')

parser.add_argument('-e', '--epoch',
                    type=int,
                    default=1,
                    help='dimension of RNN hidden states')

parser.add_argument('-minvc', '--min-vocab-count',
                    type=int,
                    default=50,
                    help='Minimum word count to include in vocab')

# Hyperparameters
parser.add_argument('-hd', '--hidden-dim',
                    type=int,
                    default=256,
                    help='dimension of RNN hidden states')

parser.add_argument('-ed', '--emb-dim',
                    type=int,
                    default=128,
                    help='dimension of word embeddings')

parser.add_argument('-bs', '--batch-size',
                    type=int,
                    default=10,
                    help='minibatch size')  # was 16. Changed to 10. Divisor of 2000

parser.add_argument('-mes', '--max-enc-steps',
                    type=int,
                    default=400,
                    help='max timesteps of encoder (max source text tokens)')

parser.add_argument('-maxds', '--max-dec-steps',
                    type=int,
                    default=10,
                    help='max timesteps of decoder (max summary tokens)')

parser.add_argument('-minds', '--min-dec-steps',
                    type=int,
                    default=35,
                    help='Minimum sequence length of generated summary. Applies only for beam search decoding mode')

parser.add_argument('-lr', '--learning-rate',
                    type=float,
                    default=0.15,
                    help='learning rate')

parser.add_argument('-ruim', '--rand-unif-init-mag',
                    type=float,
                    default=0.02,
                    help='magnitude for lstm cells random uniform inititalization')

parser.add_argument('-agic', '--adagrad-init-acc',
                    type=float,
                    default=0.1,
                    help='initial accumulator value for Adagrad')

parser.add_argument('-tnis', '--trunc-norm-init-std',
                    type=float,
                    default=1e-4,
                    help='std of trunc norm init, used for initializing everything else')

parser.add_argument('-mgn', '--max-grad-norm',
                    type=float,
                    default=2.0,
                    help='for gradient clipping')
                    
parser.add_argument('-mrp', '--load-model-path',
                    default='./saved_models',
                    help='Path to saved models to load.')

parser.add_argument('-lm', '--load-model',
                    default=True,
                    help='Path to saved models to load.')

parser.add_argument('-mrp', '--load-model-path',
                    default='./saved_models',
                    help='Path to saved models to load.')

parser.add_argument('-D', '--debug', action='store_const',
                    help='Print debug messages', dest='logging_level',
                    default=logging.INFO, const=logging.DEBUG)
"""


class MyArgs(namedtuple('args', ('data_path', 'tweet_vocab', 'hashtag_vocab', 'mode', 'log_root', 'model_root', 'epoch', 'min_vocab_count',
                    'hidden_dim', 'emb_dim', 'batch_size', 'max_enc_steps', 'max_dec_steps', 'min_dec_steps',
                    'learning_rate', 'rand_unif_init_mag', 'adagrad_init_acc', 'trunc_norm_init_std', 'max_grad_norm',
                    'debug', 'logging_level', 'load_model', 'model_num', 'load_model_path'))):
    pass


if __name__ == "__main__":
    # args = parser.parse_args()

    args = MyArgs(data_path='./tokenized_tweets4/',
                  tweet_vocab='./tokenized_tweets4/tweets-4.json.tweet_vocab.txt',
                  hashtag_vocab='./tokenized_tweets4/tweets-4.json.hashtag_vocab.txt',
                  mode='train',
                  log_root='./logs/{}'.format(datetime.now()),
                  model_root='./models',
                  load_model_path='./saved_models',
                  load_model=True,
                  model_num=231,
                  epoch=1,
                  min_vocab_count=50,
                  hidden_dim=256,
                  emb_dim=128,
                  batch_size=10,
                  max_enc_steps=400,
                  max_dec_steps=10,
                  min_dec_steps=35,
                  learning_rate=0.15,
                  rand_unif_init_mag=0.02,
                  adagrad_init_acc=0.1,
                  trunc_norm_init_std=1e-4,
                  max_grad_norm=2.0,
                  debug=logging.INFO,
                  logging_level=logging.INFO)

    main(args)
