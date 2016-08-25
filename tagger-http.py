#!/usr/bin/env python

import os
import time
import json
from urlparse import urlparse, parse_qs
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from tagger.loader import prepare_sentence
from tagger.utils import create_input, iobes_iob, iob_ranges, zero_digits
from tagger.model import Model


PORT_NUMBER = os.getenv('PORT_NUMBER', 7890)


class TaggerHttpHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        line = query_components['q'][0] if 'q' in query_components and len(query_components['q']) > 0 else ""
        
        words = line.rstrip().split()
        
        if len(words) == 0:
            self.send_response(200)
            return
        
        start = time.time()
        
        # Lowercase sentence
        if tagger.parameters['lower']:
            line = line.lower()
        # Replace all digits with zeros
        if tagger.parameters['zeros']:
            line = zero_digits(line)
        # Prepare input
        sentence = prepare_sentence(words, tagger.word_to_id, tagger.char_to_id,
                                    lower=tagger.parameters['lower'])
        input = create_input(sentence, tagger.parameters, False)
        # Decoding
        if tagger.parameters['crf']:
            y_preds = np.array(tagger.f_eval(*input))[1:-1]
        else:
            y_preds = tagger.f_eval(*input).argmax(axis=1)
        y_preds = [tagger.model.id_to_tag[y_pred] for y_pred in y_preds]
        # Output tags in the IOB2 format
        if tagger.parameters['tag_scheme'] == 'iobes':
            y_preds = iobes_iob(y_preds)
        # Write tags
        assert len(y_preds) == len(words)
        
        delimiter = '__'
        # iob = ' '.join('%s%s%s' % (w, delimiter, y)
        #                 for w, y in zip(words, y_preds))
        
        result = {
            "text":    ' '.join(words),
            "words":   words,
            "ranges":  iob_ranges(y_preds),
            # "iob":     iob,
        }
        
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("X-Time-Tagger", time.time() - start)
        self.end_headers()
        self.wfile.write(json.dumps(result))



class Tagger:
    
    def load_model(self):
        # Load existing model
        print "Loading model..."
        self.model = Model(model_path="./tagger/models/english")
        self.parameters = self.model.parameters
        
        # Load reverse mappings
        self.word_to_id, self.char_to_id, self.tag_to_id = [
            {v: k for k, v in x.items()}
            for x in [self.model.id_to_word, self.model.id_to_char, self.model.id_to_tag]
        ]
        
        # Load the model
        _, self.f_eval = self.model.build(training=False, **self.parameters)
        self.model.reload()
        print "Ready"



tagger = Tagger()



if __name__ == '__main__':
    tagger.load_model()
    server_address = ('', PORT_NUMBER)
    httpd = HTTPServer(server_address, TaggerHttpHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()


