#!/bin/env python3
# -*- coding: utf-8 -*-

from src.classifier import Classifier
from src.correios   import Tracker

import re
import csv
import random
import datetime

from nltk.tokenize import sent_tokenize

# A few global data paths
TRAINING_SET   = "data/training.csv"
GREETINGS_SET  = "data/greetings.txt"
COMPLAIN_SET   = "data/complain.txt"
HELP_SET       = "data/help.txt"
UNKNOWN_SET    = "data/unknown.txt"
WRONG_CODE_SET = "data/wrong_code.txt"
TRACKING_SET   = "data/tracking.txt"
RESULTS_SET    = "data/results.txt"
ATTEMPTS_SET   = "data/attempts.txt"
QUIT_SET       = "data/quit.txt"
GOODBYE_SET    = "data/goodbye.txt"

class Chatbot:
    
    # Regular expressions for parsing responses
    RE_CODE       = re.compile(r'.*(\w\w\d{9}\w\w)')
    RE_SENT       = re.compile(r'.*postado.*')
    RE_FORWARDED  = re.compile(r'.*encaminhado.*')
    RE_DELIVERING = re.compile(r'.*saiu.*')
    RE_ARRIVED    = re.compile(r'.*entregue.*')

    # Retries when waiting for an action
    MAX_ATTEMPTS  = 2
    
    def __init__(self):
        # Classifier
        self.c = Classifier()
        self.c.load_classifier(TRAINING_SET)
        self.code = ''

        # Correios tracker
        self.t = Tracker()

        # Answers
        with open(GREETINGS_SET) as f:
            self.greetings_responses = f.readlines()
        with open(COMPLAIN_SET) as f:
            self.complain_responses = f.readlines()
        with open(COMPLAIN_SET) as f:
            self.complain_responses = f.readlines()
        with open(HELP_SET) as f:
            self.help_responses = f.readlines()
        with open(UNKNOWN_SET) as f:
            self.unknown_responses = f.readlines()
        with open(WRONG_CODE_SET) as f:
            self.wrong_code_responses = f.readlines()
        with open(TRACKING_SET) as f:
            self.tracking_responses = f.readlines()
        with open(QUIT_SET) as f:
            self.quit_responses = f.readlines()
        with open(GOODBYE_SET) as f:
            self.goodbye_responses = f.readlines()
        with open(RESULTS_SET) as f:
            results_responses = f.readlines()
            self.offline_response    = results_responses[0]
            self.fail_response       = results_responses[1]
            self.sent_response       = results_responses[2]
            self.forwarded_response  = results_responses[3]
            self.delivering_response = results_responses[4]
            self.arrived_response    = results_responses[5]

        # If it is expecting something
        self.waiting_for = ''
        self.attempts    = 0

    # Get text intent
    def get_intent(self,text):
        return self.c.classify(text)[0]
        
    # Return a random response from a set of responses
    def random_response(self, responses_set):
        return responses_set[random.randrange(len(responses_set))]

    # Return a response with formatted results
    def format_responses(self, code):
        info = self.t.track_latest(code)
        answer = ''
        status = ''

        if info['return'] == 'request_failed':
            answer = self.offline_response
        elif info['return'] == 'failure':
            answer = self.fail_response
        elif self.RE_SENT.match(info['status']):
            answer = self.sent_response
            answer = answer.format(code, info['when'])
        elif self.RE_FORWARDED.match(info['status']):
            answer = self.forwarded_response
            answer = answer.format(info['from'], info['to'])
        elif self.RE_DELIVERING.match(info['status']):
            answer = self.delivering_response
            answer = answer.format(info['where'])
        elif self.RE_ARRIVED.match(info['status']):
            answer = self.arrived_response
        return answer

    # Test if received a tracking code and respond
    def code_responses(self, text):
        # Respond to code input
        response = self.random_response(self.unknown_responses)

        # Try to parse a code
        test = self.RE_CODE.match(text)
        if test:
            # Return the results
            self.attempts = 0
            self.waiting_for = ''
            response = self.format_responses(test.group(0))
        else:
            # If failed respond
            if self.attempts == self.MAX_ATTEMPTS:
                # Quit after MAX_ATTEMPTS
                self.attempts = 0
                self.waiting_for = ''
                response = self.random_response(self.quit_responses)
            else:
                # Else, retry
                self.attempts = self.attempts + 1
                response = self.random_response(self.wrong_code_responses)
        return response
    
    # Get a response according to an intent
    def get_response(self, text):
        # Test for tracking code
        if self.waiting_for == 'code':
            return self.code_responses(text)
        
        # Else, get responses to chat
        intent = self.get_intent(text)
        if intent == 'T':
            self.waiting_for = 'code'
        responses = {
            'G': self.greetings_responses,
            'H': self.help_responses,
            'T': self.tracking_responses,
            'C': self.complain_responses,
            'U': self.unknown_responses,
            'B': self.goodbye_responses
        }.get(intent, 'U')
        return self.random_response(responses)

def main():
    # A few examples
    c = Chatbot()
    print(c.get_response('oi'))
    print(c.get_response('tchau'))
    print(c.get_response('me ajuda por favor'))
    print(c.get_response('quero rastrear um pacote'))
    print(c.get_response('A'))
    print(c.get_response('B'))
    print(c.get_response('C'))
    print(c.get_response('quero rastrear um pacote'))
    print(c.get_response('A'))
    print(c.get_response('DW144693630BR'))
    print(c.get_response('meu pacote está atrasado'))
    print(c.get_response('te amo'))

if __name__ == "__main__":
    main()
