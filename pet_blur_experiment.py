import experiment_functions as ef
from pathlib import Path
import argparse
import time

if __name__ == "__main__":

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
    a = parser.parse_args()

    images = Path(a.images).glob("*.jpg")

    experiments = ef.BlurExperiments(a.fid_m, a.hd_m)

    for image in images:

        image_name = Path(image).resolve().stem
        # cache_pickle_name = "./results/{}_data.pkl".format(adv_image_name)

        if a.use_cache is not 'y':

            print("\n--------Processing img number:{}----------\n".format(image_name))
            start = time.time()
            full_df = experiments.blur_iter_experiment(str(image),
                                                       str(image),
                                                       a.blur_iter,
                                                       a.hd_thres,
                                                       a.mfd,
                                                       a.blur_kernel,
                                                       a.blur_size, True)
            end = time.time()

            print("Time elapsed: {} seconds".format(end - start))
            print("Saving data to Pickle format")
            image_name = Path(image).resolve().stem
            full_df.to_pickle("./results/{}_{}_data.pkl".format(image_name, a.blur_kernel))

            print("\n--------Processing finished----------\n")
