# Author: Anwesh Tuladhar <anwesh.tuladhar@gmail.com>
# Website: https://anwesht.github.io/
__author__ = 'Anwesh Tuladhar'


import tensorflow as tf
from tensorflow.python import debug as tf_debug

import numpy as np

import os
import time
from collections import namedtuple
import json
import queue


# ----------------------------------------------
# Setup script arguments
# These are common hyperparameter settings.
# ----------------------------------------------
FLAGS = tf.app.flags.FLAGS

# Input data params
tf.app.flags.DEFINE_string('data_path', './tokenized', 'Path folder with tokenized json files.')
tf.app.flags.DEFINE_string('tweet_vocab', './tokenized/tweets.tweet_vocab.txt',
                           'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('hashtag_vocab', './tokenized/tweets.hashtag_vocab.txt',
                           'Path expression to hashtag vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')

# Output params
tf.app.flags.DEFINE_string('log_root', './logs', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '',
                           'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 10, 'minibatch size')  # was 16. Changed to 10. Divisor of 2000
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000,
                            'Size of vocabulary. These will be read from the vocabulary file in order. '
                            'If the vocabulary file contains fewer words than this number, '
                            'or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('restore_best_model', False,
                            'Restore the best model in the eval/ dir and save it in the train/ dir, '
                            'ready to be used for further training. Useful for early stopping, or if your training '
                            'checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")


# ---------
# Constants
# ---------
START_HASHTAG = '<h>'
END_HASHTAG = '</h>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'


class Vocab:
    def __init__(self, vocab_file):
        """
        Class to track vocabulary used in tweets and hashtags

        Args:
            vocab_file: Path to vocab file
        """
        self._wordToId = dict()
        self._idToWord = dict()

        self._id = 0
        # create id's for special tokens.
        for t in [PAD_TOKEN, UNKNOWN_TOKEN, START_HASHTAG, END_HASHTAG]:
            self._wordToId[t] = self._id
            self._idToWord[self._id] = t
            self._id += 1

        tf.logging.info('Loading vocab file: {}'.format(vocab_file))

        with open(vocab_file, 'r') as vf:
            for line in vf:
                try:
                    w, _ = line.split()
                except ValueError as e:
                    tf.logging.warn('Skipping... Error unpacking vocab: \"{}\" in file: {}'.format(line.strip(),
                                                                                                   vocab_file))
                    continue

                self._wordToId[w] = self._id
                self._idToWord[self._id] = w
                self._id += 1

        tf.logging.info("Finished creating vocabulary object. Num words = {}".format(self._id))

    def word2id(self, w):
        if w in self._wordToId:
            return self._wordToId[w]
        return self._wordToId[UNKNOWN_TOKEN]

    def id2word(self, i):
        if i in self._idToWord:
            return self._idToWord[i]

        tf.logging.error("Id {} not in vocabulary".format(i))
        raise KeyError("Id {} not in vocabulary".format(i))

    def size(self):
        return self._id


class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, tweet, hashtags, tweet_vocab, hashtags_vocab, hps):
        """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target
        sequences, which are stored in self.

        Args:
          tweet: source text; a string. each token is separated by a single space.
          hashtags: list of strings, one per abstract sentence. In each sentence, each token is separated by
            a single space.
          tweet_vocab: Tweet Vocabulary object
          hashtags_vocab: Hashtag Vocabulary object
          hps: hyperparameters
        """
        self.hps = hps

        # Get ids of special tokens
        start_decoding = hashtags_vocab.word2id(START_HASHTAG)
        stop_decoding = hashtags_vocab.word2id(END_HASHTAG)

        # Process the tweet
        tweet_words = tweet.split()
        self.enc_len = len(tweet_words)  # store the length of the tweet before padding
        self.enc_input = [tweet_vocab.word2id(w) for w in tweet_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the hashtags
        hashtag_words = hashtags.split()  # list of strings
        hashtags_ids = [hashtags_vocab.word2id(w) for w in hashtag_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(hashtags_ids, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # Store the original strings
        self.original_tweet = tweet
        self.original_hashtag = hashtags

    @staticmethod
    def get_dec_inp_targ_seqs(hashtags, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target
        sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len.
        The input sequence must start with the start_id and the target sequence must end with the stop_id
        (but not if it's been truncated).

        Args:
          hashtags: List of ids (integers)
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + hashtags[:]
        target = hashtags[:]
        target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target


    # Used in Batch.init_decoder_seq
    def pad_decoder_inp_target(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)

        while len(self.target) < max_len:
            self.target.append(pad_id)

    # Used in Batch.init_encoder_seq
    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)


# TODO: Implement this. DONE
class Batch:
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, tweet_vocab, hashtag_vocab):
        """Turns the example_list into a Batch object.

        Args:
           example_list: List of Example objects
           hps: hyperparameters
           vocab: Vocabulary object
        """
        self.pad_id = tweet_vocab.word2id(PAD_TOKEN) # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
        self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
        self.store_orig_strings(example_list) # store the original strings

    def init_encoder_seq(self, example_list, hps):
        """Initializes the following:
            self.enc_batch:
              numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
            self.enc_lens:
              numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
            self.enc_padding_mask:
              numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

          If hps.pointer_gen, additionally initializes the following:
            self.max_art_oovs:
              maximum number of in-article OOVs in the batch
            self.art_oovs:
              list of list of in-article OOVs (strings), for each example in the batch
            self.enc_batch_extend_vocab:
              Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        # self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        # self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        # self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)
        self.encoder_inputs = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.encoder_input_lengths = np.zeros((hps.batch_size), dtype=np.int32)
        self.encoder_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.encoder_inputs[i, :] = ex.enc_input[:]
            self.encoder_input_lengths[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.encoder_padding_mask[i][j] = 1

            # print('Initializing Encoder sequence: {:2f}% Complete. '.format(100.0 * i / len(example_list)), end='\r', flush=True)

        # tf.logging.info('Finished initializing Encoder sequence.')


    def init_decoder_seq(self, example_list, hps):
        """Initializes the following:
        self.dec_batch:
          numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
        self.target_batch:
          numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
        self.dec_padding_mask:
          numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
        """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_target(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
        # self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        # self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        # self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
        self.decoder_inputs = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.decoder_outputs = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.decoder_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        # at: Adding for TraningHelper: decoder_lengths
        self.decoder_output_lengths = np.zeros((hps.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.decoder_inputs[i, :] = ex.dec_input[:]
            self.decoder_outputs[i, :] = ex.target[:]
            self.decoder_output_lengths[i] = ex.dec_len

            for j in range(min(ex.dec_len, 10)):
                self.decoder_padding_mask[i][j] = 1

            # print('Initializing Decoder sequence: {:2f}% Complete. '.format(100.0 * i / len(example_list)), end='\r', flush=True)

        # tf.logging.info('Finished initializing Decoder sequence.')


    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        self.original_tweets = [ex.original_tweet for ex in example_list]  # list of lists
        self.original_hashtags = [ex.original_hashtag for ex in example_list]  # list of lists


class Batcher:
    # XXX: hardcoded
    TWEETS_PER_FILE = 2000  # Number of tweets per file.

    def __init__(self, filepath, tweet_vocab, hashtag_vocab, hps):
        self._filepath = filepath
        self._tweet_vocab = tweet_vocab
        self._hashtag_vocab = hashtag_vocab

        self._hps = hps
        # max number of batches the batch_queue can hold ==> TWEETS_PER_FILE / batch_size
        # self._batch_queue = queue.Queue(self.TWEETS_PER_FILE // hps.batch_size)
        # Queue blocks if max size is reached.
        self._batch_queue = queue.Queue()

        self.create_training_batches()

    def next(self):
        """Return a Batch from the batch queue.

        If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

        Returns:
          batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        tf.logging.info('Retrieving next batch from batcher.')
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize()
            )
            return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    """
    def next(self):
        with open(self._filepath, 'r') as f:
            batch_size = self._hps.batch_size
            examples_list = list()

            for line in f:
                # tweet, hashtags, tweet_vocab, hashtags_vocab, hps
                obj = json.loads(line)
                tweet = obj['text']
                hashtags = obj['hashtags']

                # yield Example(tweet, hashtags, self._tweet_vocab, self._hashtag_vocab)
                examples_list.append(Example(tweet, hashtags, self._tweet_vocab, self._hashtag_vocab))

                if len(examples_list) == batch_size:
                    cur_batch = Batch(examples_list, self._hps, self._tweet_vocab, self._hashtag_vocab)
                    examples_list = list()
                    yield cur_batch

        # Reach here when no more examples. Return None so that a new batcher is created.
        tf.logging.warn('Out of examples in current batcher. {}'.format(self._filepath))
        yield None
    """

    #  TODO: Created training batches. REMANING: CREATE DECODE BATCHES.
    def create_training_batches(self):
        """
        Takes Examples out of example queue, sorts them by encoder sequence length,
        processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        """
        tf.logging.info('Creating training batch for file: {}'.format(self._filepath))
        t1 = time.time()

        all_examples = list()

        with open(self._filepath, 'r') as f:
            for i, line in enumerate(f):
                # tweet, hashtags, tweet_vocab, hashtags_vocab, hps
                obj = json.loads(line)
                tweet = obj['text']
                hashtags = obj['hashtags']

                all_examples.append(Example(tweet, hashtags, self._tweet_vocab, self._hashtag_vocab, self._hps))

        t2 = time.time()
        tf.logging.info('Creating training batch: Finished reading file in {} seconds.'.format(t2-t1))

        all_examples = sorted(all_examples, key=lambda inp: inp.enc_len)

        all_batches = list()
        batch_size = self._hps.batch_size

        num_batches = len(all_examples) // batch_size

        for i in range(0, len(all_examples), batch_size):
            all_batches.append(all_examples[i:i+batch_size])

        for i, b in enumerate(all_batches):
            self._batch_queue.put(Batch(b, self._hps, self._tweet_vocab, self._hashtag_vocab))
            print('Creating batch queue: {:2f}% Complete. '.format(100.0 * (i+1) / len(all_batches)), end='\r', flush=True)

        t3 = time.time()

        tf.logging.info('Created training batch in: {} seconds. Total time taken: {}'.format(t3-t2, t3-t1))

        # --------------------------------------

        """
        while True:
            if self._hps.mode != 'decode':
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in xrange(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in xrange(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            else:  # beam search decode mode
                ex = self._example_queue.get()
                b = [ex for _ in xrange(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))
                
        """


class Model:
    def __init__(self, hps, tweet_vocab, hashtag_vocab):
        """

        Args:
            hps (NamedTuple): Hyper-parameters to use for the model
            tweet_vocab (Vocab): Instance of class Vocab for tweets
            hashtag_vocab (Vocab): Instance of class Vocab for hashtags
        """
        self.hps = hps
        self.tweet_vocab = tweet_vocab
        self.hashtag_vocab = hashtag_vocab

    def _add_placeholders(self):
        """ Add input placeholders to the tensorflow graph """
        hps = self.hps
        # TODO: Put comments on the tensor shape
        self._encoder_inputs = tf.placeholder(tf.int32,
                                              shape=[hps.batch_size, None],  # [batch size, source seq length]
                                              name='encoder_inputs')
        self._decoder_inputs = tf.placeholder(tf.int32,
                                             shape=[hps.batch_size, None],  # [batch size, target seq length]
                                             name='decoder_inputs')
        # TODO: Check if we keep None OR use hps.max_dec_steps
        self._decoder_outputs = tf.placeholder(tf.int32,
                                               shape=[hps.batch_size, None],  # [batch size, target seq length]
                                               name='decoder_outputs')

        # Placeholder for sequence lengths of each src input.
        # This is used in dynamic rnn
        self._encoder_input_lengths = tf.placeholder(tf.int32,
                                                     shape=[hps.batch_size],  # [batch size]
                                                     name='encoder_input_lengths')

        # Adding for decoder_lengths. used in TrainingHelper
        self._decoder_output_lengths = tf.placeholder(tf.int32,
                                                     shape=[hps.batch_size],  # [batch size]
                                                     name='decoder_output_lengths')

        # Target weights???
        self._decoder_padding_mask = tf.placeholder(tf.float32,
                                                    shape=[hps.batch_size, hps.max_dec_steps],
                                                    name='decoder_padding_mask')

        # Encoder padding mask???
        self._encoder_padding_mask = tf.placeholder(tf.float32,
                                                    shape=[hps.batch_size, None],
                                                    name='encoder_padding_mask')

    def _add_seq2seq(self):
        """
        Add embeddings layer, encoder rnn and decoder rnn to the tensorflow graph
        Embeddings layer: One set of embedding weights per language are learned during training.
        """
        hps = self.hps
        src_vocab_size = self.tweet_vocab.size()

        # Add embedding layer
        with tf.variable_scope("embedding"):
            # embedding for source tweet vocab
            self._encoder_embedding = tf.get_variable(
                name='encoder_embedding',
                shape=[src_vocab_size, hps.emb_dim]  # [src vocab size, emb dim]
            )

            # Lookup embedding for current input batch
            self._encoder_emb_inputs = tf.nn.embedding_lookup(
                params=self._encoder_embedding,
                ids=self._encoder_inputs  # [batch size, source seq length]
            )

            # embedding for dec tweet vocab
            #   _encoder_emb_inputs: [batch size, source seq length, emb_dim]  # TODO: verify
            #   _encoder_inputs: [batch size, source seq length]
            self._decoder_embedding = tf.get_variable(
                name='decoder_embedding',
                shape=[src_vocab_size, hps.emb_dim]  # [src vocab size, emb dim]
            )

            # Lookup embedding for decoder input batch
            #   _decoder_emb_inputs: [batch size, target seq length, emb_dim]  # TODO: verify
            #   _decoder_inputs: [batch size, target seq length]
            self._decoder_emb_inputs = tf.nn.embedding_lookup(
                params=self._decoder_embedding,
                ids=self._decoder_inputs  # [batch size, target seq length]
            )

        # Build the encoder
        with tf.variable_scope("encoder") as encoder_scope:
            # Build encoder RNN cell
            # at: This is deprecated. Replacing with new api.
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hps.hidden_dim)
            # encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=hps.hidden_dim, name='basic_lstm_cell')

            # Run Dynamic RNN
            #   encoder_outputs: [batch size, src seq length, hidden_dim]
            #   encoder_state: [batch size, hidden_dim]
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell, inputs=self._encoder_emb_inputs,
                sequence_length=self._encoder_input_lengths,
                # at: adding because of error: "ValueError: If there is no initial_state, you must give a dtype."
                dtype=encoder_scope.dtype
            )

            # TODO: Check whether to use outputs or state. pointer generator uses output, tutorial uses state.
            self._encoder_states = encoder_outputs
            self._encoder_outputs = encoder_outputs

        with tf.variable_scope("decoder") as decoder_scope:
            # Build decoder RNN cell
            # This is deprecated. Replacing with new api.
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hps.hidden_dim)
            # decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=hps.hidden_dim, name='basic_lstm_cell')

            # Use the tensorflow seq2seq library.
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self._decoder_emb_inputs,
                # sequence_length=decoder_length, # TODO: decoder lengths ???
                sequence_length=self._decoder_output_lengths, # TODO: decoder lengths. Setup in Batch
            )

            # Decoder
            self._output_layer = tf.layers.Dense(
                units=self.hashtag_vocab.size(),
                use_bias=False,
                name='output_projection'
            )

            # at: Decoder initial state = encoder zero state?
            # see: nmt/model.py:_build_decoder_cell():843
            encoder_state = encoder_cell.zero_state(hps.batch_size, dtype=tf.float32)  # self.dtype = float32.

            decoder_initial_state = encoder_state

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                # initial_state=self._encoder_states,  # TODO: verify what to use.
                initial_state=decoder_initial_state,
                # output_layer=projection_layer  # TODO: projection layer ??? Not used in actual code. Applying it later
            )

            # Dynamic decoding
            final_outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                swap_memory=True,  # Passed to tf.while_loop
                scope=decoder_scope
            )
            logits = self._output_layer(final_outputs.rnn_output)

            # Compute Training loss
            # TODO: pointer generator uses seq2se1.sequence_loss OR mas_and_avg loss. ???
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._decoder_outputs,  # outputs placeholder [batch size, target seq length]
                logits=logits
            )

            # TODO: target_weights = zero-one matrix of the same size as decoder_outputs. It masks padding positions
            # outside of the target sequence lengths with values 0.
            # Same as dec_padding_mask in pointer generator???
            # NOTE: dividing by batch_size so that the hyperparameters are "invariant" to batch_size
            # XXX: Using decoder_padding_mask. See nmt/model.py:_compute_loss:657 for actual target_weights
            target_weights = self._decoder_padding_mask

            self._train_loss = (tf.reduce_sum(crossent * target_weights) / hps.batch_size)

            # Backpropagation pass: Gradient computation and optimization
            # Calculate and clip gradients
            tvars = tf.trainable_variables()
            # Compute the symbolic derivatives of sum of ys w.r.t. x in xs.
            #   gradients: list of tensor of length len(xs) where each tensor = sum(dy/dx) for each x in xs.
            gradients = tf.gradients(
                ys=self._train_loss,
                xs=tvars)

            # Important step in training RNNs is gradient clipping.
            # max_gradient_norm is usually set to values like 5 or 1.
            #   clipped_gradients: list of tensors of same type as list_t
            #   global_norm: 0-D tensor (scalar) representing the global norm
            clipped_gradients, global_norm = tf.clip_by_global_norm(
                t_list=gradients,
                clip_norm=hps.max_grad_norm  # TODO: ???
            )

            # Add Optimizer
            optimizer = tf.train.AdagradOptimizer(learning_rate=hps.lr,
                                                  initial_accumulator_value=hps.adagrad_init_acc)

            self._train_step = optimizer.apply_gradients(
                grads_and_vars=zip(clipped_gradients, tvars),
                global_step=self.global_step,
                name='train_step'
            )

    def build_graph(self):
        """ Build the tensorflow dataflow graph for the encoder-decoder seq2seq model """
        tf.logging.info("Building dataflow graph")
        t_start = time.time()

        # Track global training steps (i.e. the number of training batches seen by the graph.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Add input placeholders for encoder and decoder
        self._add_placeholders()

        # Add the encoder-decoder seq2seq model
        self._add_seq2seq()

        # # Track global training steps (i.e. the number of training batches seen by the graph.
        # self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Merge all summaries
        self._summaries = tf.summary.merge_all()

        t_end = time.time()
        tf.logging.info("Graph built in {} seconds.".format(t_end-t_start))

    # TODO: Implement this.
    def execute_train_step(self, sess, batch):
        """
        Run one training iteration.
        Args:
            sess: Tensorflow session
            batch: Instance of Batch

        Returns:
            Returns a dictionary containing train op, summaries, loss and global_step
        """
        # feed_dict = self._make_feed_dict(batch)
        # Make feed_dict
        feed_dict = dict()

        feed_dict[self._encoder_inputs] = batch.encoder_inputs
        feed_dict[self._encoder_input_lengths] = batch.encoder_input_lengths
        # TODO: _enc_padding_mask???
        feed_dict[self._encoder_padding_mask] = batch.encoder_padding_mask

        # if not just_enc:
        feed_dict[self._decoder_inputs] = batch.decoder_inputs
        feed_dict[self._decoder_outputs] = batch.decoder_outputs
        feed_dict[self._decoder_padding_mask] = batch.decoder_padding_mask
        # at: adding for TrainingHelper => decoder_lengths
        feed_dict[self._decoder_output_lengths] = batch.decoder_output_lengths


        to_return = {
            'train_op': self._train_step,
            # TODO. ADD SUMMARIES!!!
            # 'summaries': self._summaries,
            'loss': self._train_loss,
            'global_step': self.global_step,
        }

        # tf.logging.info('to_return: {}'.format(to_return))
        # tf.logging.info('feed_dict: {}'.format(feed_dict))

        # Run the operations.
        return sess.run(to_return, feed_dict)


def setup_training(model, hps):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    model.build_graph() # build the graph
    # if FLAGS.restore_best_model:
    #     restore_best_model()

    saver = tf.train.Saver(max_to_keep=100)  # keep 3 checkpoints at a time

    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                             save_model_secs=60,      # checkpoint every 60 secs
                             global_step=model.global_step)

    summary_writer = sv.summary_writer

    tf.logging.info("Preparing or waiting for session...")

    # TODO: util.get_config() ???
    sess_context_manager = sv.prepare_or_wait_for_session()  # config=util.get_config()
    tf.logging.info("Created session.")
    try:
        run_training(model, sess_context_manager, sv, summary_writer, hps)  # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        # sv.stop()
    finally:
        sv.stop()


def run_training(model, sess_context_manager, sv, summary_writer, hps):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    tokenized_files = [os.path.join(FLAGS.data_path, p) for p in os.listdir(FLAGS.data_path) if p.endswith('.json')]

    current_file_index = 0
    num_files = len(tokenized_files)
    batcher = Batcher(tokenized_files[current_file_index], model.tweet_vocab, model.hashtag_vocab, model.hps)

    tf.logging.info("Starting training loop...")
    with sess_context_manager as sess:
        if FLAGS.debug: # start the tensorflow debugger
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        while True:  # repeats until interrupted
            # Get next batch. If current batcher is exhausted, create new batcher.
            try:
                batch = batcher.next()
            except Exception as e:
                tf.logging.warn('End of batcher for file: {}'.format(batch._filepath))

                current_file_index = (current_file_index + 1) % num_files
                batcher = Batcher(tokenized_files[current_file_index], model._tweet_vocab, model._hashtag_vocab,
                                  model._hps)
                batch = batcher.next()

            print('Current batch is from file: {}'.format(batcher._filepath))
            tf.logging.info('Current batch is from file: {}'.format(batcher._filepath))

            tf.logging.info('running training step...')
            t0 = time.time()
            results = model.execute_train_step(sess, batch)
            t1 = time.time()
            tf.logging.info('seconds for training step: %.3f', t1-t0)

            loss = results['loss']
            tf.logging.info('loss: %f', loss) # print( the loss to screen

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            # get the summaries and iteration number so we can write summaries to tensorboard
            summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
            train_step = results['global_step'] # we need this to update our running average loss

            summary_writer.add_summary(summaries, train_step) # write the summaries
            if train_step % 100 == 0: # flush the summary writer every so often
                summary_writer.flush()


def main(unused_args):
    if len(unused_args) != 1:
        raise Exception('Undefined flags: {}'.format(unused_args))

    # Handle logging parameters
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting application in mode: {}'.format(FLAGS.mode))

    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == 'train':
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logs directory {} doesn't exist. Run in train mode to begin.".format(FLAGS.log_root))

    tweet_vocab = Vocab(FLAGS.tweet_vocab)
    hashtag_vocab = Vocab(FLAGS.hashtag_vocab)

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                   'pointer_gen']
    hps_dict = {}
    # for key, val in FLAGS.__flags.iteritems():  # for each flag
    # for key, val in FLAGS.__flags.items():  # for each flag
    for key, val in FLAGS.flag_values_dict().items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    tf.logging.info('Hyper-parameters: {}'.format(hps))

    # TODO: Create a data-batcher???
    # Batcher will be created for each file in the folder.

    # Seed randomness ???
    tf.set_random_seed(111)

    # Create Model
    model = Model(hps, tweet_vocab, hashtag_vocab)

    # Train Model OR Run Decoder.
    setup_training(model, hps)


if __name__ == "__main__":
    tf.app.run(main=main)

