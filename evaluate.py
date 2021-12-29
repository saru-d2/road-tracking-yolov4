# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Compute metrics for trackers using MOTChallenge ground-truth data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path

import motmetrics as mm


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--loglevel', type=str,
                        help='Log level', default='info')
    parser.add_argument('--fmt', type=str,
                        help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str,
                        help='LAP solver to use for matching between frames.')
    parser.add_argument('--id_solver', type=str,
                        help='LAP solver to use for ID metrics. Defaults to --solver.')
    parser.add_argument('--exclude_id', dest='exclude_id', default=False, action='store_true',
                        help='Disable ID metrics')
    parser.add_argument('--dataPath', type=str, default='')
    return parser.parse_args()


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    # print(ts)
    accs = []
    names = []
    # print("compare")
    for k, tsacc in ts.items():
        #print("K ")
        # print(k)
        # print(tsacc)
        #print(" ")
        print(k)
        if k in gts:
            logging.info('Comparing %s...', k)
            accs.append(mm.utils.compare_to_groundtruth(
                gts[k], tsacc, 'iou', distth=0.5))
            print(mm.utils.compare_to_groundtruth(
                gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)
        #print(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
        # print(k)

    return accs, names


def main():
    # pylint: disable=missing-function-docstring
    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(
        level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    gtfiles = sorted([os.path.join(args.dataPath, 'gt', 'gt', i)
               for i in os.listdir(os.path.join(args.dataPath, 'gt', 'gt'))])
    tsfiles = sorted([os.path.join(args.dataPath, 'tst', i)
               for i in os.listdir(os.path.join(args.dataPath, 'tst'))])

    logging.info('Found %d groundtruths and %d test files.',
                 len(gtfiles), len(tsfiles))
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    gt = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0],
                     mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0],
                     mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])
    # print(gt)
    # print(ts)
    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)
    motchallenge_metric_names = {
        'idf1': 'IDF1',
        'idp': 'IDP',
        'idr': 'IDR',
        'recall': 'Rcll',
        'precision': 'Prcn',
        'num_unique_objects': 'GT',
        'mostly_tracked': 'MT',
        'partially_tracked': 'PT',
        'mostly_lost': 'ML',
        'num_false_positives': 'FP',
        'num_misses': 'FN',
        'num_switches': 'IDs',
        'num_fragmentations': 'FM',
        'mota': 'MOTA',
        'motp': 'MOTP',
        'num_transfer': 'IDt',
        'num_ascend': 'IDa',
        'num_migrate': 'IDm'

    }
    metrics = list(motchallenge_metric_names)
    if args.exclude_id:
        metrics = [x for x in metrics if not x.startswith('id')]

    logging.info('Running metrics')

    if args.id_solver:
        mm.lap.default_solver = args.id_solver
    summary = mh.compute_many(
        accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters,
          namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')


if __name__ == '__main__':
    main()
