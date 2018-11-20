# Author: Anwesh Tuladhar <anwesh.tuladhar@gmail.com>
# Website: https://anwesht.github.io/
__author__ = 'Anwesh Tuladhar'


import tensorflow as tf

import os
import time
from collections import namedtuple


# -----------------------
# Setup script arguments
# -----------------------
FLAGS = tf.app.flags.FLAGS

# Input data params
tf.app.flags.DEFINE_string('data_path', './tokenize', 'Path folder with tokenized json files.')
tf.app.flags.DEFINE_string('tweet_vocab', './tokenize/tweets.tweet_vocab.txt',
                           'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('hashtag_vocab', './tokenize/tweets.hashtag_vocab.txt',
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
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
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
        for t in [UNKNOWN_TOKEN, START_HASHTAG, END_HASHTAG]:
            self._wordToId[t] = self._id
            self._idToWord[self._id] = t

        with open(vocab_file, 'r') as vf:
            for line in vf:
                w, _ = line.split()
                self._wordToId[w] = self._id
                self._idToWord[self._id] = w

                self._id += 1

        tf.logging.info("Finished creating vocabulary object. Num words = {}".format(_id))

    def word2id(self, w):
        if w in self._wordToId:
            return self._wordToId[w]
        return self._wordToId[UNKNOWN_TOKEN]

    def word2id(self, i):
        if i in self._idToWord:
            return self._idToWord[i]

        tf.logging.error("Id {} not in vocabulary".format(i))
        raise KeyError("Id {} not in vocabulary".format(i))

    def size(self):
        return self._id


class Model:
    def __init__(self, hps, tweet_vocab, hashtag_vocab):
        """

        Args:
            hps (NamedTuple): Hyper-parameters to use for the model
            tweet_vocab (Vocab): Instance of class Vocab for tweets
            hashtag_vocab (Vocab): Instance of class Vocab for hashtags
        """
        self._hps = hps
        self._tweet_vocab = tweet_vocab
        self._hashtag_vocab = hashtag_vocab

    def _add_placeholders(self):
        """ Add input placeholders to the tensorflow graph """
        hps = self._hps
        # TODO: Put comments on the tensor shape
        self._encoder_inputs = tf.placeholder(tf.int32,
                                              shape=[hps.batch_size, None],  # [batch size, source seq length]
                                              name='encoder_inputs')
        self._decoder_inputs= tf.placeholder(tf.int32,
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

        # Target weights???
        self._decoder_padding_mask = tf.placeholder(tf.int32,
                                                    shape=[hps.batch_size, hps.max_dec_steps])

    def _add_seq2seq(self):
        """
        Add embeddings layer, encoder rnn and decoder rnn to the tensorflow graph
        Embeddings layer: One set of embedding weights per language are learned during training.
        """
        hps = self._hps
        src_vocab_size = self._tweet_vocab.size()

        # Add embedding layer
        with tf.variable_scope("embedding"):
            # embedding for source tweet vocab
            self._encoder_embedding = tf.get_variable(
                name='encoder_embedding',
                shape=[src_vocab_size, hps.emb_dim]  # [src vocab size, emb dim]
            )

            # Lookup embedding for current input batch
            self._encoder_emb_inputs = tf.nn.embedding_lookup(
                parmas=self._encoder_embedding,
                ids=self._encoder_inputs  # [batch size, source seq length]
            )

            # embedding for dec tweet vocab
            #   _encoder_emb_inputs: [batch size, source seq length, emb_dim]  # TODO: verify
            #   _encoder_inputs: [batch size, source seq length]
            self._decoder_embedding = tf.get_variable(
                name='encoder_embedding',
                shape=[src_vocab_size, hps.emb_dim]  # [src vocab size, emb dim]
            )

            # Lookup embedding for decoder input batch
            #   _decoder_emb_inputs: [batch size, target seq length, emb_dim]  # TODO: verify
            #   _decoder_inputs: [batch size, target seq length]
            self._decoder_emb_inputs = tf.nn.embedding_lookup(
                parmas=self._decoder_embedding,
                ids=self._decoder_inputs  # [batch size, target seq length]
            )

        # Build the encoder
        with tf.variable_scope("encoder"):
            # Build encoder RNN cell
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hps.hidden_dim)

            # Run Dynamic RNN
            #   encoder_outputs: [batch size, src seq length, hidden_dim]
            #   encoder_state: [batch size, hidden_dim]
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell, inputs=self._encoder_emb_inputs,
                sequence_length=self._encoder_input_lengths
            )

            # TODO: Check whether to use outputs or state. pointer generator uses output, tutorial uses state.
            self._encoder_states = encoder_outputs

        with tf.variable_scopr("decoder") as decoder_scope:
            # Build decoder RNN cell
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hps.hidden_dim)

            # Use the tensorflow seq2seq library.
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self._decoder_emb_inputs,
                sequence_length=decoder_length, # TODO: decoder lengths ???
            )

            # Decoder
            self._output_layer = tf.layers.Dense(
                units=self._hashtag_vocab.size(),
                use_bias=False,
                name='output_projection'
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=helper,
                initial_state=self._encoder_states,
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
            train_loss = (tf.reduce_sum(crossent * target_weights) / hps.batch_size)

            # Backpropagation pass: Gradient computation and optimization
            # Calculate and clip gradients
            tvars = tf.trainable_variables()
            # Compute the symbolic derivatives of sum of ys w.r.t. x in xs.
            #   gradients: list of tensor of length len(xs) where each tensor = sum(dy/dx) for each x in xs.
            gradients = tf.gradients(
                ys=train_loss,
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
                grads_and_vars=(clipped_gradients, tvars),
                global_step=self.global_step,
                name='train_step'
            )

    def build_graph(self):
        """ Build the tensorflow dataflow graph for the encoder-decoder seq2seq model """
        tf.logging.info("Building dataflow graph")
        t_start = time.time()
        # Add input placeholders for encoder and decoder
        self._add_placeholders()

        # Add the encoder-decoder seq2seq model
        self._add_seq2seq()

        # Track global training steps (i.e. the number of training batches seen by the graph.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Merge all summaries
        self._summaries = tf.summary.merge_all()

        t_end = time.time()
        tf.logging.info("Graph built in {} seconds.".format(t_end-t_start))


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
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # TODO: Create a data-batcher???

    # Seed randomness ???
    tf.set_random_seed(111)

    # Create Model
    model = Model(hps, tweet_vocab, hashtag_vocab)
    
    # Train Model OR Run Decoder.






if __name__ == "__main__":
    tf.app.run(main=main)

