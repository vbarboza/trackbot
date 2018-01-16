#!/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import requests
import traceback
from flask import Flask, request

from src.chatbot import Chatbot

# Facebook Graph API Messages URL
FB_GRAPH = 'https://graph.facebook.com/v2.6/me/messages/?access_token='

# Server access token
token = os.environ.get('FB_ACCESS_TOKEN')

# Flask app
app = Flask(__name__)

# Chatbot
c = Chatbot()

# Simple reponses to GET and POST
@app.route('/', methods=['GET', 'POST'])
def webhook():

    if request.method == 'POST':
        # Respond to POST messages
        try:
            # Read input
            data = json.loads(request.data.decode())
            text = data['entry'][0]['messaging'][0]['message']['text']
            sender = data['entry'][0]['messaging'][0]['sender']['id']
            # Make a response
            payload = {'recipient':{'id':sender},
                       'message':{'text':c.get_response(text)}}
            # Send a response
            r = requests.post(FB_GRAPH + token, json=payload)
            return r
        except Exception as e:
            # CRASH
            print(traceback.format_exc())
    elif request.method == 'GET':
        if request.args:
            if request.args.get('hub.verify_token') == os.environ.get('FB_VERIFY_TOKEN'):
                # Success
                return request.args.get('hub.challenge')
            # Failure
            return 'Failed!'
        return 'Hello, world!'
    
if __name__ == '__main__':
    app.run(debug=True)
