"""
Extract a subset in JSON from a dump file.

Usage:
    extract-dump.py [options] DUMP

Options:
    DUMP
        Process dump file DUMP
    --verbose
        Turn on verbose logging.
"""

import logging
import sys
from os import fspath
from pathlib import Path
import subprocess as sp
import json
import bz2
import lzma
from docopt import docopt

import mwxml
from tqdm import tqdm

_log = logging.getLogger('extract-dump')


def load_dump(dump_file: Path):
    _log.info('reading %s', dump_file)
    with bz2.open(dump_file) as bzf:
        dump = mwxml.Dump.from_file(bzf)
        for page in dump:
            if page.namespace == 0 and not page.redirect:
                rev = next(page)
                text = rev.text
                yield page.id, page.title, text


def process_dump(dump_file: Path):
    stem = dump_file.name.replace('.xml.bz2', '')
    json_file = dump_file.parent / f'{stem}.json.zstd'
    _log.info('saving to %s', json_file)

    child = sp.Popen(['zstd', '-9', '-o', fspath(json_file)], stdin=sp.PIPE)
    jsf = child.stdin

    for id, title, text in tqdm(load_dump(dump_file)):
        obj = {'id': id, 'title': title, 'text': text}
        jsf.write(json.dumps(obj).encode('utf8'))
        jsf.write(b'\n')
    
    jsf.close()
    result = child.wait()
    result.check_returncode()


def main(opts):
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

    file = Path(opts['DUMP'])
    process_dump(file)



if __name__ == '__main__':
    opts = docopt(__doc__)
    main(opts)
