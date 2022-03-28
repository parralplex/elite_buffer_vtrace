import matplotlib.pyplot as plt
import numpy as np
import os

from collections import deque
from itertools import zip_longest


def set_global_chart_settings():
    plt.rc('axes', titlesize=27)
    plt.rc('axes', labelsize=27)
    plt.rc('xtick', labelsize=26)
    plt.rc('ytick', labelsize=26)
    plt.rc('legend', fontsize=25)
    plt.rc('figure', titlesize=27)

    plt.rcParams["figure.figsize"] = (13, 8)


def create_chart(data_file_name_list, experiment_url_list, x_line_index_list, y_line_index_list, curve_labels_list, x_label, y_label, chart_file_name, avg_buff_size):
    plt.clf()
    for i in range(len(data_file_name_list)):
        os.chdir(experiment_url_list[i])
        root_dir = os.getcwd()

        data_files = []
        data = []

        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(data_file_name_list[i]):
                    data_files.append(open(filepath, "r"))
                    data.append(deque())

        MEAN_THRESHOLD = avg_buff_size
        scores_file_data = [[[], []] for i in range(len(data))]

        for line_counter, line_data in enumerate(zip_longest(*data_files)):
            for counter, number in enumerate(line_data):
                if number is not None:
                    num = float(number.split(',')[y_line_index_list[i]].rstrip('\n'))

                    data[counter].append(num)
                    if len(data[counter]) > MEAN_THRESHOLD:
                        data[counter].popleft()

                    scores_file_data[counter][0].append(float(number.split(',')[x_line_index_list[i]].rstrip('\n')))
                    scores_file_data[counter][1].append(np.average(data[counter]))

        x_min = min(min(data_file[0]) for data_file in scores_file_data)
        x_max = max(max(data_file[0]) for data_file in scores_file_data)

        x = np.linspace(x_min, x_max, 200)

        interpolated = [np.interp(x, data_file[0], data_file[1]) for data_file in scores_file_data]

        avg_y = [np.average(y_approximate) for y_approximate in zip(*interpolated)]

        upper_err_y = [np.average(y_approximate) + np.std(y_approximate) for y_approximate in zip(*interpolated)]
        lower_err_y = [np.average(y_approximate) - np.std(y_approximate) for y_approximate in zip(*interpolated)]

        plt.plot(x, avg_y, label=curve_labels_list[i])
        plt.fill_between(x, lower_err_y, upper_err_y, alpha=0.15)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig(chart_file_name, transparent=True)
