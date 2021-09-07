import matplotlib.pyplot as plt
from collections import deque
import numpy as np


def set_global_chart_settings():
    plt.figure(figsize=(10.0, 8.0))

    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=20)


def create_chart(folder_url, data_file_name, x_label, y_label, curve_labels, chart_file_name, avg_buff_size, calc_avg=True, skip_first_rows=0):
    plt.clf()
    data_file = open(folder_url + "/" + data_file_name, "r")

    x = []
    y = []
    avg_buff = []
    first_line = True
    for i, line in enumerate(data_file):
        if i < skip_first_rows:
            continue
        line_data = line.split(',')
        line_data[len(line_data) - 1] = line_data[len(line_data) - 1].rstrip('\n')
        line_sum = 0
        for j in range(len(line_data)):
            if first_line:
                y.append([])
                avg_buff.append(deque())
            if calc_avg:
                avg_buff[j].append(float(line_data[j]))
                if len(avg_buff[j]) > avg_buff_size:
                    avg_buff[j].popleft()
                y[j].append(np.average(avg_buff[j]))
            else:
                y[j].append(float(line_data[j]))
            line_sum += float(line_data[j])
            if len(line_data) > 1 and j == (len(line_data)-1):
                if first_line:
                    y.append([])
                    avg_buff.append(deque())
                if calc_avg:
                    avg_buff[j+1].append(line_sum)
                    if len(avg_buff[j+1]) > avg_buff_size:
                        avg_buff[j+1].popleft()
                    y[j+1].append(np.average(avg_buff[j+1]))
                else:
                    y[j + 1].append(line_sum)
        first_line = False
        x.append(i)

    for i in range(len(y)):
        st_dv = np.std(y[i])
        plt.plot(x, y[i], label=curve_labels[i])
        plt.fill_between(x, y[i]-st_dv, y[i]+st_dv, alpha=0.5)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(folder_url + "/Charts/" + chart_file_name, transparent=True)
