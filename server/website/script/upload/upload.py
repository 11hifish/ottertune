#
# OtterTune - upload.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import argparse
import logging
import os
import urllib2
from poster.encode import multipart_encode
from poster.streaminghttp import register_openers

register_openers()

# Logging
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)


def upload(datadir, upload_code, server):
    params = {
        'summary': open(os.path.join(datadir, 'sample-0__summary.json'), "r"),
        'knobs': open(os.path.join(datadir, 'sample-0__knobs.json'), "r"),
        'metrics_start': open(os.path.join(datadir, 'sample-0__metrics_start.json'), 'r'),
        'metrics_end': open(os.path.join(datadir, 'sample-0__metrics_end.json'), 'r'),
        'upload_code': upload_code,
    }

    datagen, headers = multipart_encode(params)

    request = urllib2.Request(server + "/new_result/", datagen, headers)

    LOG.info(urllib2.urlopen(request).read())


def main():
    parser = argparse.ArgumentParser(description="Upload generated data to the website")
    parser.add_argument('datadir', type=str, nargs=1,
                        help='Directory containing the generated data')
    parser.add_argument('upload_code', type=str, nargs=1,
                        help='The website\'s upload code')
    parser.add_argument('server', type=str, default='http://0.0.0.0:8000',
                        nargs='?', help='The server\'s address (ip:port)')
    args = parser.parse_args()
    upload(args.datadir[0], args.upload_code[0], args.server)


if __name__ == "__main__":
    main()
