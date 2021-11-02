#!/usr/bin/env python3


import os
import json
import logging
import numpy
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import pyplot


def main():
    arg_parser = ArgumentParser(description="Generate output digest from a design experiments directory.")
    arg_parser.add_argument("-o", "--output", type=Path, default=Path("/tmp/design-results.png"), help="Output graph.")
    arg_parser.add_argument("resultsdir", type=Path, help="The results directory containing all the experiments.")
    args = arg_parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("looking for parameters and accuracy files...")
    param_files = [
        (Path(base_dir)/'params.json', Path(base_dir)/'accuracy.json')
        for base_dir, _dirs, files in os.walk(args.resultsdir)
        if {'params.json', 'accuracy.json'} <= set(files)
    ]

    logging.info("found %d experiments, collecting them", len(param_files))
    experiments = defaultdict(lambda: defaultdict(list))
    for params_file, acc_file in param_files:
        params = json.loads(params_file.read_text())
        experiments[params['design']][params['coeff']].append(json.loads(acc_file.read_text())['accuracy'])

    logging.info("averaging")
    experiments = {
        design: sorted(
            {
                coeff: sum(acc_list)/len(acc_list)
                for coeff, acc_list in params_info.items()
            }.items(),
            key=lambda param_pair: param_pair[0], # sort by coeff
        )
        for design, params_info in experiments.items()
    }

    logging.info("to numpy")
    experiments = {
        design: (
            numpy.array([coeff for coeff, _ in samples]),
            numpy.array([sample for _, sample in samples]),
        )
        for design, samples in experiments.items()
    }

    logging.info("plotting!")
    fig, axis = pyplot.subplots(nrows=1, ncols=1, figsize=(16, 12), dpi=80)
    axis.set_title("Total Loss")
    for design, (coeff, samples) in experiments.items():
        logging.info("    plotting %s", design)
        axis.plot(coeff, samples, label=design)
    fig.legend(loc="upper right")
    fig.savefig(args.output)

if __name__ == "__main__":
    main()
