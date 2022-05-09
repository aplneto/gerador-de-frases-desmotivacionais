# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:20:43 2022

@author: aplne
"""
import json
import re
from flask import Flask, request, Response
from flask_cors import cross_origin
from model import default_config_dict, model, tokenizer, get_completions

HOST = '0.0.0.0'
PORT = '80'
NAME = ''

app = Flask('')

@app.route('/get_raw_sentences', methods=['POST'])
@cross_origin()
def get_raw_sentences():
    content_type = request.content_type
    if (content_type != 'application/json'):
        return Response(
            json.dumps({'status': 400, 'msg': 'Bad Request'}),
            status=400,
            mimetype='application/json'
        )
    content = request.json
    kw = content.get('keywords', [])
    settings = content.get('settings', default_config_dict)
    print(kw)
    s = get_completions(
        kw, tokenizer=tokenizer, model=model, **settings
    )
    return Response(json.dumps(s), status=200, mimetype='application/json')

@app.route('/get_sentences', methods=['POST'])
@cross_origin()
def get_sentences():
    content_type = request.content_type
    if (content_type != 'application/json'):
        return Response(
            json.dumps({'status': 400, 'msg': 'Bad Request'}),
            status=400,
            mimetype='application/json'
        )
    content = request.json
    kw = content.get('keywords', [])
    settings = content.get('settings', default_config_dict)
    print(kw)
    samples = get_completions(
        kw, tokenizer=tokenizer, model=model, **settings
    )
    sentences = []
    for sample in samples:
        sentence = re.split(r'[\.\!\?]', sample)
        if len(sentence) > 1:
            sentences.append(sentence[1])
        else:
            sentences.append(sample)
    return Response(
        json.dumps(sentences), status=200, mimetype='application/json'
    )


@app.route('/', methods=['GET'])
@cross_origin()
def index():
    if request.args.get('6843f1fffdc2bf82a7dc08584717b008', None) is not None:
        return Response(status=200)

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
