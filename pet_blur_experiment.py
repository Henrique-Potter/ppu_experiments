import experiment_functions as ef
from pathlib import Path
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fid_m", type=str, required=True)
    parser.add_argument("--hd_m", type=str, required=True)
    parser.add_argument("--img_b", type=str, required=True)
    parser.add_argument("--img_t", type=str, required=True)
    parser.add_argument("--blur_iter", type=int, required=True)
    parser.add_argument("--hd_thres", type=float, required=True)
    parser.add_argument("--mfd", type=bool, required=True)
    parser.add_argument("--blur_kernel", type=str, required=True)
    parser.add_argument("--blur_size", type=int, required=True)
    a = parser.parse_args()

    images = Path('./images').glob("*.jpg")

    experiments = ef.BlurExperiments(a.fid_m, a.hd_m)

    for image in images:
        full_df = experiments.blur_iter_experiment(str(image),
                                                   str(image),
                                                   a.blur_iter,
                                                   a.hd_thres,
                                                   a.mfd,
                                                   a.blur_kernel,
                                                   a.blur_size, True)

        print(full_df)

        if full_df is not None:
            image_name = Path(image).resolve().stem
            full_df.to_pickle("./results/{}_data.pkl".format(image_name))
        else:
            pass


