import experiment_functions as ef
from pathlib import Path
import pandas as pd
import argparse
import time
import os


def process_image(image_name, image, args, experiment):

    print("\n--------Processing img number:{}----------\n".format(image_name))
    start = time.time()
    full_df = experiment.blur_iter_experiment(str(image),
                                              str(image),
                                              args.blur_iter,
                                              args.hd_thres,
                                              args.mfd,
                                              args.blur_kernel,
                                              args.blur_size, True)

    end = time.time()
    print("Time elapsed: {} seconds".format(end - start))
    print("Saving data to Pickle format")
    full_df.to_pickle("./results/{}_{}_data.pkl".format(image_name, args.blur_kernel))
    print("\n--------Processing finished----------\n")
    return full_df


def main_method():

    parser = argparse.ArgumentParser()
    parser.add_argument("--fid_m", type=str, required=True)
    parser.add_argument("--hd_m", type=str, required=True)
    parser.add_argument("--img_b", type=str, required=True)
    parser.add_argument("--img_t", type=str, required=True)
    parser.add_argument("--blur_iter", type=int, required=True)
    parser.add_argument("--hd_thres", type=float, required=True)
    parser.add_argument("--mfd", type=str, required=True)
    parser.add_argument("--blur_kernel", type=str, required=True)
    parser.add_argument("--blur_size", type=int, required=True)
    parser.add_argument("--use_cache", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)

    args = parser.parse_args()
    images = Path(args.images).glob("*.jpg")
    experiment = ef.BlurExperiments(args.fid_m, args.hd_m)

    for image in images:

        image_name = Path(image).resolve().stem
        cache_pickle_name = "./results/{}_{}_data.pkl".format(image_name, args.blur_kernel)

        if args.use_cache == 'y':

            if os.path.exists(cache_pickle_name):
                pass
            else:
                process_image(image_name, image, args, experiment)

        else:
            process_image(image_name, image, args, experiment)


if __name__ == "__main__":

    main_method()
