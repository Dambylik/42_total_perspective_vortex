import sys
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt

N_SUBJECTS = 109 # dataset contains 109 participants
EXPERIMENTS = [
    [3, 7, 11],               # 0: real movement,     left fist vs right fist
    [4, 8, 12],               # 1: imagined movement, left fist vs right fist
    [5, 9, 13],               # 2: real movement,     both fists vs both feet
    [6, 10, 14],              # 3: imagined movement, both fists vs both feet
    [3, 7, 11, 4, 8, 12],     # 4: real + imagined,   left fist vs right fist
    [5, 9, 13, 6, 10, 14],    # 5: real + imagined,   both fists vs both feet
]
DATA_PATH = "./data"


def load_raw(subject, runs):
    files = eegbci.load_data(subject, runs, path=DATA_PATH, update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.plot_sensors(show_names=True)
    plt.show(block=True)
    return raw


# def get_epochs(subject, runs):
#     raw = filter_raw(load_raw(subject, runs))


# def evaluate(subject, runs):
#     X_train, y_train = get_epochs(subject, runs[:-1])
#     X_test, y_test = get_epochs(subject, runs[-1:])
#     return build_pipeline


# def run_all():
#     means = []
#     for exp, runs in enumerate(EXPERIMENTS):
#         scores = []
#         for subject in range(1, N_SUBJECTS + 1):
#             try:
#                 score = evaluate(subject, run)




def main():
    # args = sys.args[1:]
    # if not args:
    #     return run_all()
    # if len(args) != 3 or args[2] not in ["train", "predict"] or not args[0].isdigit() or not args[1].isdigit():
    #     sys.exit(__doc__)
    # subject, run, mode = int(args[0]), int(args[1]), args[2]
    # if not 1 <= subject <= N_SUBJECTS:
    #     sys.exit(f"ubject must be between 1 and {N_SUBJECTS}")
    # try:
    #     find_experiment(run)
    # except ValueError as error:
    #     sys.exit(str(error))
    # train(subject, run) if mode == "train" else predict(subject, run)
    load_raw(1, 3)


if __name__ == "__main__":
    main()