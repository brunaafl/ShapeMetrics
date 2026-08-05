
import ray
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import plot_time_corrs, plot_all_regions, load_and_process_fold

def main(args):

    w, s = 10, 2

    if len(args.region) == 1:
        region = args.region[0]

        logger.info("Submitting response")
        all_corrs_resp = load_and_process_fold(region, 'response', w, s)
        logger.info("Submitting stimulus")
        all_corrs_stim = load_and_process_fold(region, 'stim', w, s)

        # Save
        np.savez(f'slided_time_corrs_{region}.npz', all_corrs_resp=all_corrs_resp, all_corrs_stim=all_corrs_stim)
        
        logger.info("Finished loading. Plotting results.")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
        all_corrs = {'response': all_corrs_resp, 'stim': all_corrs_stim}
        plot_time_corrs(all_corrs, ax=ax, w=w, s=s, save_path=f'slided_time_corrs.svg')
        logger.info("Finished plotting results.")

    else:
        all_corrs = {}
        for region in args.region:
            # For the individual regions, just doing it for the stimulus to make the plot a bit cleaner
            logger.info(f"Submitting stimulus for region: {region}")
            all_corrs_stim = load_and_process_fold(region, 'stim', w=w, s=s)
            all_corrs[region] = all_corrs_stim

        plot_all_regions(all_corrs, w=w, s=s, save_path='time_corrs_all_regions_stim.svg')
        logger.info("Finished plotting results.")
    #ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=list, default=['all'])
    args = parser.parse_args()
    main(args)