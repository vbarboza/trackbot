#!/bin/env python3
# -*- coding: utf-8 -*-

import re
import requests
from unidecode import unidecode


class Tracker:

    # Third-party WEBSRO service
    SERVICE_URL = 'http://www.websro.com.br/detalhes.php?P_COD_UNI={}'

    # Regular expressions for parsing the source
    RE_DATE = re.compile(r'.*<td rowspan="2">(\d\d/\d\d/\d\d\d\d \d\d:\d\d)</td>')
    RE_STATUS = re.compile(r'.*<td colspan="2"><strong>(.*)</strong></td>')
    RE_PLACE  = re.compile(r'.*<td colspan="2">Local: (.*)</td>')
    RE_FROM   = re.compile(r'.*<td>Origem: (.*)</td>')
    RE_TO     = re.compile(r'.*<td>Destino: (.*)</td>')

    # Parsing states
    ST_MATCH_INIT          = 0
    ST_MATCH_DATE          = 1
    ST_MATCH_STATUS        = 2
    ST_MATCH_PLACE         = 3
    ST_MATCH_DESTINATION   = 4
    ST_MATCH_COMPLETE      = 5

    # Return the latest tracking information
    def track_latest(self, code):
        return self.track_history(code, only_latest = True)[0]

    # Return the tracking history
    def track_history(self, code, only_latest = False):
        
        # Get the page
        r = requests.get(self.SERVICE_URL.format(code), stream = True)

        # Test the HTTP status code
        if r.status_code != requests.codes.ok:
            return [{'return':'request_failed'}]

        # Parse the page
        history = []
        state = 0
        for line in r.iter_lines():
            if line:
                line = line.decode('utf-8')
                # Initializing
                if state == self.ST_MATCH_INIT:
                    when = ''
                    status = ''
                    where = ''
                    where_from = ''
                    where_to = ''
                    state = self.ST_MATCH_DATE

                # Looking for a date
                if state == self.ST_MATCH_DATE:
                    test = self.RE_DATE.match(line)
                    if test:
                        when = test.group(1)
                        state = self.ST_MATCH_STATUS

                # Looking for tracking information/status
                if state == self.ST_MATCH_STATUS:
                    test = self.RE_STATUS.match(line)
                    if test:
                        status = test.group(1)
                        state = self.ST_MATCH_PLACE

                # Looking for a place or origin
                if state == self.ST_MATCH_PLACE:
                    test = self.RE_PLACE.match(line)
                    if test:
                        # If a place is found without an origin, finishes here
                        where = test.group(1)
                        state = self.ST_MATCH_COMPLETE
                    else:
                        # Else, continue parsing
                        test = self.RE_FROM.match(line)
                        if test:
                            where_from = test.group(1)
                            state = self.ST_MATCH_DESTINATION

                # Looking for a destination
                if state == self.ST_MATCH_DESTINATION:
                    test = self.RE_TO.match(line)
                    if test:
                        where_to = test.group(1)
                        state = self.ST_MATCH_COMPLETE

                # Creating a dict with the results
                if state == self.ST_MATCH_COMPLETE:
                    info = {}
                    info['return']    = 'success'
                    info['date']      = unidecode(when)
                    info['status']    = unidecode(status)
                    if where:
                        info['place'] = unidecode(where)
                    else:
                        info['from']  = unidecode(where_from)
                        info['to']    = unidecode(where_to)

                    # Push to the front of the history
                    history.insert(0, info)
                    # Stop it here if we don't need the history
                    if only_latest:
                        break
                    state = self.ST_MATCH_INIT

        # If no information found, return a 'failure'
        return history if history else [{'return':'failure'}]

def main():
    # A few examples
    t = Tracker()
    print(t.track_history('')) 
    print(t.track_history('DW144693630BR'))
    print(t.track_latest('DW144693630BR'))
    
if __name__ == '__main__':
    main()
