
import ray
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import plot_time_corrs, load_and_process_fold

def main(args):

    w, s = 15, 2

    if len(args.region) == 1:
        region = args.region[0]

        logger.info("Submitting response")
        all_corrs_resp = load_and_process_fold(region, 'response', w, s)
        logger.info("Submitting stimulus")
        all_corrs_stim = load_and_process_fold(region, 'stim', w, s)
        
        logger.info("Finished loading. Plotting results.")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
        all_corrs = {'response': all_corrs_resp, 'stim': all_corrs_stim}
        plot_time_corrs(all_corrs, ax=ax, w=w, s=s, save_path=f'time_corrs_{region}.svg')
        logger.info("Finished plotting results.")

    else:
        all_corrs = {}
        for region in args.region:
            logger.info(f"Submitting response for region: {region}")
            all_corrs_resp = load_and_process_fold(region, 'response', w, s)
            logger.info(f"Submitting stimulus for region: {region}")
            all_corrs_stim = load_and_process_fold(region, 'stim', w, s)
            
            all_corrs[region] = {'response': all_corrs_resp, 'stim': all_corrs_stim}
    
    #ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=list, default=["all"])
    args = parser.parse_args()
    main(args)