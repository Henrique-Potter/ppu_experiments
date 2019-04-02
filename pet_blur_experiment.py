import experiment_functions as ef
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

    blur_iters, fm_results, hd_results = ef.blur_iter_experiment(a.fid_m,
                                                                 a.hd_m,
                                                                 a.img_b,
                                                                 a.img_t,
                                                                 a.blur_iter,
                                                                 a.hd_thres,
                                                                 a.mfd,
                                                                 a.blur_kernel,
                                                                 a.blur_size, True)
