from pi_face_detection import PiFaceDet
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fid_m", type=str, required=True)
    # parser.add_argument("--hd_m", type=str, required=True)
    # parser.add_argument("--hd_thres", type=float, required=True)
    parser.add_argument("--preview", type=bool, required=True)

    args = parser.parse_args()

    peyes = PiFaceDet()
    peyes.run_identification(5)
    peyes.run_learn_face(5)
    peyes.run_identification(5)
