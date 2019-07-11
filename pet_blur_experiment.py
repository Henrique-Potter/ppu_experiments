from concurrent.futures.thread import ThreadPoolExecutor
import experiment_functions as ef
from multiprocessing.dummy import Pool as ThreadPool
from ai import object_detection as hd
from ai import face_match as fm
from pathlib import Path
import argparse
import time
import os

show_debug_window = False


def process_image(image_name, image, args, experiment, results_path):

    print("\n--------Processing img number:{}----------\n".format(image_name))
    start = time.time()

    full_df = experiment.blur_iter_experiment(str(image),
                                              str(image),
                                              args.blur_iter,
                                              args.hd_thres,
                                              args.mfd,
                                              args.blur_kernel,
                                              args.blur_size, show_debug_window)

    end = time.time()
    print("Time elapsed: {} seconds".format(end - start))
    print("Saving data to Pickle format")
    print("\n--------Processing finished----------\n")

    pool.apply_async(cache_data, args=(args, full_df, image_name, results_path))

    return full_df


def cache_data(args, full_df, image_name, results_path):
    full_df.to_pickle(results_path + "/{}_{}_data.pkl".format(image_name, args.blur_kernel))


def main_method():

    parser = argparse.ArgumentParser()
    parser.add_argument("--fid_m", type=str, required=True)
    parser.add_argument("--hd_m", type=str, required=True)
    parser.add_argument("--blur_iter", type=int, required=True)
    parser.add_argument("--hd_thres", type=float, required=True)
    parser.add_argument("--mfd", type=str, required=True)
    parser.add_argument("--blur_kernel", type=str, required=True)
    parser.add_argument("--blur_size", type=int, required=True)
    parser.add_argument("--use_cache", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)

    args = parser.parse_args()
    images = Path(args.images).glob("*.jpg")

    face_det = fm.FaceMatch(args.fid_m)
    human_det = hd.CocoDetectorAPI(path_to_ckpt=args.hd_m)

    experiment = ef.BlurExperiments(face_det, human_det)

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)

    for image in images:

        image_name = Path(image).resolve().stem
        full_image_path = args.results + "/{}_{}_data.pkl".format(image_name, args.blur_kernel)

        if args.use_cache == 'y' and os.path.exists(full_image_path):
            pass
        else:
            process_image(image_name, image, args, experiment, args.results)


if __name__ == "__main__":

    pool = ThreadPool(4)
    executor = ThreadPoolExecutor(max_workers=8)
    main_method()
