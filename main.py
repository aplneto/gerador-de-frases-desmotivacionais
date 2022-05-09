# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:20:43 2022

@author: aplne
"""
import json
import re
from flask import Flask, request, Response
from model import default_config_dict, model, tokenizer, get_completions

HOST = '0.0.0.0'
PORT = '8000'
NAME = ''

app = Flask('')

@app.route('/get_raw_sentences', methods=['POST'])
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
    print(kw)
    s = get_completions(
        kw, tokenizer=tokenizer, model=model, **default_config_dict
    )
    return Response(json.dumps(s), status=200, mimetype='application/json')

@app.route('/get_sentences', methods=['POST'])
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
    print(kw)
    samples = get_completions(
        kw, tokenizer=tokenizer, model=model, **default_config_dict
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

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)