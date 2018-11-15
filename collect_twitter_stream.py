#!/usr/bin/env python
# vim: set sw=4 sts=4 expandtab:
# author: Giovanni Luca Ciampaglia <gciampag@indiana.edu>

# TODO: set request timeout to 90 seconds to manage stalls and implement
# backoff strategies as documented here:
# https://dev.twitter.com/streaming/overview/connecting (see Stalls and
# Reconnecting)
#
# and also see:
# http://docs.python-requests.org/en/latest/user/quickstart/#timeouts

""" Use the stream API to track tweets """

from __future__ import print_function

try:
    import ujson as json
except ImportError:
    import json
import sys
import requests
import logging
import time
import gzip
import socket
from requests_oauthlib import OAuth1
from argparse import ArgumentParser, FileType

from datetime import datetime, timedelta


url = 'https://stream.twitter.com/1.1/statuses/filter.json'
woeid = 23424977
trends_url = 'https://api.twitter.com/1.1/trends/place.json'
trending_wait_hrs = 1
log_fmt = '%(asctime)s: %(threadName)s: %(levelname)s: %(message)s'

# all times are in seconds. Start with time, then depending on whether
# kind is linear or exponential either add step or multiply by factor.
# Raise a fatal error after waiting max.
backoff_params = {
    'tcp': {
        'time': 0,
        'kind': 'linear',
        'step': .250,
        'max': 16
    },
    'http': {
        'time': 5,
        'kind': 'exponential',
        'factor': 2,
        'max': 320
    },
    'http_420': {
        'time': 60,
        'kind': 'exponential',
        'factor': 2,
        'max': 600
    }
}


class TwitterStreamError(Exception):
    pass


class TwitterStream(object):

    def __init__(self, f, outpath, params):
        if "track_trending" in params:
            logging.info("track-trending: {track_trending}""".format(**params))
            self._track_trending = params['track_trending']
            self._trending_timeout = datetime.now() + timedelta(hours=trending_wait_hrs)
            del params['track_trending']
        else:
            self._track_trending = None

        self.outpath = outpath
        self.params = params
        self.credentials = json.load(f)
        self._stall_timeout = 90
        self._backoff_sleep = None  # if not None, we are backing off
        self._backoff_strategy = None
        self._conn_timeout_sleep = .5
        self._counter = 0  # overall counter of processed tweets
        self._backoff_params = backoff_params
        logging.info("appending to: {}".format(self.outpath))
        logging.info("follow: {follow}".format(**params))
        logging.info("track: {track}".format(**params))
        logging.info("locations: {locations}".format(**params))
        logging.info("stall_warnings: {stall_warnings}".format(**params))
        logging.info("delimited: {delimited}""".format(**params))

    def _update_trends_and_backoff(self, strategy):
        if self._track_trending and datetime.now() >= self._trending_timeout:
           # self._update_trends(self.params)
            self._update_trends()
        else:
            self._backoff(strategy)

    def _update_trends(self):
        now = datetime.now()
        logging.info("updating trends at: {}".format(now))

        try:
            response = self.client.get(trends_url, params={'id': self._track_trending})
        except AttributeError as e:
            logging.error("Trying to update trends without client object. Raising fatal error!")
            raise AttributeError

        if response.ok:
            # at: worked in mac.
            #trends = json.loads(response.content)
            trends = json.loads(response.content.decode('utf-8'))
            trends = set([trend['name'] for trend in trends[0]['trends']])
            hash_tags = set()

            for t in trends:
                if t.startswith('#'):
                    hash_tags.add(t[1:])

            self.params['track'] = ','.join(hash_tags)
            self._trending_timeout = now + timedelta(hours=trending_wait_hrs)

            logging.info("new trends to track: {}".format(self.params['track']))
            logging.info("next trend update after: {}".format(self._trending_timeout))
        else:
            logging.warning("Could not update trends at: {}".format(now))

    def _append(self):
        if self.outpath == '-':
            self.outfp = sys.stdout
        elif self.outpath.endswith("gz"):
            self.outfp = gzip.GzipFile(filename=self.outpath, mode='a')
        else:
            self.outfp = open(self.outpath, 'ab')  # at: was 'a'

    def _authenticate(self):
        """
        Authenticate and return a requests client object.
        """
        c = self.credentials
        oauth = OAuth1(client_key=c['consumer_key'],
                       client_secret=c['consumer_secret'],
                       resource_owner_key=c['access_token'],
                       resource_owner_secret=c['access_token_secret'],
                       signature_type='auth_header')
        self.client = requests.session()
        self.client.auth = oauth

    def _backoff(self, strategy):
        """
        See https://dev.twitter.com/streaming/overview/connecting (Stalls and Reconnecting)

        A strategy defines a set of parameters for the backoff, including the initial time,
        the way it increases the sleep period (linear or exponential), and a
        maximum time after which it's better to just raise a fatal error.
        """
        try:
            params = self._backoff_params[strategy]
        except KeyError:
            raise ValueError("unknown strategy: {}".format(strategy))
        if self._backoff_sleep is None or self._backoff_strategy != strategy:
            # start with initial time if first backoff or if strategy has changed
            self._backoff_sleep = params['time']
            self._backoff_strategy = strategy
        else:
            # continue with previous strategy
            if self._backoff_sleep >= params['max']:
                logging.error("Reached maximum backoff time. Raising fatal error!")
                raise TwitterStreamError()
            if params['kind'] == 'linear':
                self._backoff_sleep += params['step']
            else:
                self._backoff_sleep *= params['factor']
        # at: Moving outside.
        logging.warning("Sleeping {:.2f}s as part of {} backoff.".format(self._backoff_sleep, params['kind']))
        time.sleep(self._backoff_sleep)

    def _reset_backoff(self):
        self._backoff_sleep = None
        self._backoff_strategy = None

    def stream(self):
        logging.info("Started streaming.")
        try:
            if self._track_trending:
                self._authenticate()
                self._update_trends()

            while True:
                try:
                    self._append()
                    self._authenticate()
                    stream = self.client.post(url, data=self.params, stream=True, timeout=self._stall_timeout)
                    data_lines = 0  # includes keepalives
                    for line in stream.iter_lines():
                        data_lines += 1
                        if line:
                            self.outfp.write(line + b'\n')
                            self._counter += 1
                            if self._counter % 1000 == 0:
                                logging.info("{} tweets.".format(self._counter))
                        # at: seems like trends is not being updated enough times. Calling it here as well.
                        if self._track_trending and datetime.now() >= self._trending_timeout:
                            self._update_trends()
                        if data_lines >= 8:
                            # reset backoff status if received at least 8 data
                            # lines (including keep-alive newlines). Stream
                            # seems to send at least 8 keepalives, regardless
                            # of whether authentication was successful or not.
                            logging.debug("Reset backoff")
                            self._reset_backoff()
                            data_lines = 0
                    logging.warning("Backing off..")
                    # self._backoff('tcp')
                    self._update_trends_and_backoff('tcp')
                except requests.exceptions.ConnectTimeout:
                    # wait just a (small) fixed amount of time and try to
                    # reconnect.
                    msg = "Timeout while trying to connect to server. Retrying in {}s.."
                    logging.warning(msg.format(self._conn_timeout_sleep))
                    time.sleep(self._conn_timeout_sleep)
                except requests.Timeout:
                    # catching requests.Timeout instead of requests.ReadTimeout
                    # because we are setting a timeout parameter in the POST
                    msg = "Server did not send any data for {}s. Backing off.."
                    logging.warning(msg.format(self._stall_timeout))
                    self._backoff('tcp')
                except requests.ConnectionError:
                    logging.warning("Reconnecting to stream endpoint...")
                    self._backoff('tcp')
                except socket.error as e:
                    msg = "Socket error {}: {}. Reconnecting to stream endpoint..."
                    logging.warning(msg.format(e.errno, e.message))
                    self._backoff('tcp')
                except requests.HTTPError as e:
                    if e.response.status_code == 420:
                        msg = "Got HTTP 420 Error. Backing off.."
                        logging.warning(msg)
                        self._backoff("http_420")
                    else:
                        msg = "Got HTTP Error. Backing off.."
                        logging.warning(msg)
                        self._backoff("http")
                except KeyboardInterrupt:
                    logging.info("got ^C from user. Exit.")
                    return
        finally:
            # catch any fatal error (including TwitterStreamError we raise if
            # backoff reaches maximum sleep time)
            stream.close()
            try:
                self.outfp.close()
            except IOError as e:
                if e.errno == 32:  # Broken Pipe
                    pass


parser = ArgumentParser(description=__doc__)
parser.add_argument("key_file", metavar="key", help="key JSON file",
                    type=FileType())
parser.add_argument("output_path", metavar="output",
                    help="output file (append mode)")
parser.add_argument("-f", "--follow",
                    help="comma-separated list of user IDs")
parser.add_argument("-t", "--track",
                    help="keywords to track")
# at: adding track trending
parser.add_argument("-tt", "--track-trending",
                    help="woeid of the location to track trending hash tags for. (eg: US => 23424977)")
parser.add_argument("-l", "--locations",
                    help="set of bounding boxes")
parser.add_argument("-d", "--delimited", action="store_const",
                    const="length")
parser.add_argument("-s", "--stall-warnings", action="store_const",
                    const="true")
parser.add_argument('-D', '--debug', action='store_const',
                    help='print debug messages', dest='logging_level',
                    default=logging.INFO, const=logging.DEBUG)

if __name__ == '__main__':
    args = parser.parse_args()
    #logging.basicConfig(format=log_fmt, level=args.logging_level)
    logging.basicConfig(format=log_fmt, level=args.logging_level,
                        filename="output.nov15.log", filemode='a')
    if (args.follow is None) \
            and (args.track is None) \
            and (args.locations is None) \
            and (args.track_trending is None):
        parser.error("please specify at least one of follow/track/track_trending/locations")
    params = dict(args._get_kwargs())
    del params['key_file']
    del params['output_path']
    del params['logging_level']
    streamer = TwitterStream(args.key_file, args.output_path, params)
    # at: log any exception.
    try:
        streamer.stream()
    except Exception as e:
        logging.error("Exiting script: \n {}".format(e))

