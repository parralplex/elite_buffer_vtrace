import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os

# plt.fill_between(x, y-st_dv, y+st_dv, alpha = 0.5)    # error plotting

def create_charts(folder_url):
    os.mkdir(folder_url + "/Charts")
    score_file = open(folder_url + "/Scores.txt", "r")

    x = []
    y = []
    y1 = []
    scores = deque([])
    for i, line in enumerate(score_file):
        line = float(line.rstrip('\n'))
        scores.append(line)
        if len(scores) > 100:
            scores.popleft()
        y.append(np.average(scores))
        y1.append(line)
        x.append(i)

    plt.clf()
    plt.figure(figsize=(8.0, 6.0))

    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    plt.rc('figure', titlesize=20)

    plt.plot(x, y1, alpha=0.5, label="raw_data")
    plt.plot(x, y, color="red", label="Avg(100)")

    plt.xlabel('Episodes')
    plt.ylabel('Reward per episode')
    plt.legend()
    plt.savefig(folder_url + "/Charts/reward_chart.png", transparent=True)
    # plt.show()

    plt.clf()

    episode_step_file = open(folder_url + "/Episode_steps.txt", "r")

    x = []
    y = []
    y1 = []
    steps = deque([])
    for i, line in enumerate(episode_step_file):
        line = float(line.rstrip('\n'))
        steps.append(line)
        if len(steps) > 100:
            steps.popleft()
        y.append(np.average(steps))
        y1.append(line)
        x.append(i)

    plt.plot(x, y1, alpha=0.5, label="raw_data")
    plt.plot(x, y, color="red", label="Avg(100)")

    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.legend()
    plt.savefig(folder_url + "/Charts/episode_steps_chart.png", transparent=True)

    plt.clf()

    max_reward_file = open(folder_url + "/max_reward_file.txt", "r")

    x = []
    y = []
    steps = deque([])
    for i, line in enumerate(max_reward_file):
        line = float(line.rstrip('\n'))
        steps.append(line)
        if len(steps) > 100:
            steps.popleft()
        y.append(np.average(steps))
        x.append(i)

    plt.plot(x, y, color="red", label="max reward")

    plt.xlabel('Episodes')
    plt.ylabel('Max reward')
    plt.legend()
    plt.savefig(folder_url + "/Charts/max_reward_chart.png", transparent=True)

    plt.clf()

    loss_file = open(folder_url + "/Loss_file.txt", "r")

    x = []
    y = []
    y1 = []
    y2 = []
    y3 = []
    yqueue = deque()
    y1queue = deque()
    y2queue = deque()
    y3queue = deque()
    for i, line in enumerate(loss_file):
        if i < 50:
            continue
        a, b, c = line.split(',')
        c = c.rstrip('\n')
        yqueue.append(float(a))
        if len(yqueue) > 100:
            yqueue.popleft()
        y.append(np.average(yqueue))
        y1queue.append(float(b))
        if len(y1queue) > 100:
            y1queue.popleft()
        y1.append(np.average(y1queue))
        y2queue.append(float(c))
        if len(y2queue) > 100:
            y2queue.popleft()
        y2.append(np.average(y2queue))
        y3queue.append(float(a) + float(b) + float(c))
        if len(y3queue) > 100:
            y3queue.popleft()
        y3.append(np.average(y3queue))
        x.append(i)

    plt.plot(x, y1, label="baseline")
    plt.plot(x, y, color="red", label="policy")
    plt.plot(x, y2, label="entropy")
    plt.plot(x, y3, label="total")

    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(folder_url + "/Charts/loss_chart.png", transparent=True)

    plt.clf()
    power_draw_file = open(folder_url + "/Power_draw.txt", "r")

    x = []
    y = []
    yqueue = deque()
    for i, line in enumerate(power_draw_file):
        a, b, c = line.split(',')
        c = c.rstrip('\n')
        yqueue.append(float(a) + float(b) + float(c))
        if len(yqueue) > 100:
            yqueue.popleft()
        y.append(np.average(yqueue))
        x.append(i)

    plt.plot(x, y, color="red", label="avg(100)")

    plt.xlabel('Seconds')
    plt.ylabel('Power draw in wats')
    plt.legend()
    plt.savefig(folder_url + "/Charts/power_draw_chart.png", transparent=True)

    plt.clf()

    plt.figure(figsize=(10.0, 6.0))
    lr_file = open(folder_url + "/lr_file.txt", "r")

    x = []
    y = []
    y1 = []
    steps = deque([])
    for i, line in enumerate(lr_file):
        line = float(line.rstrip('\n'))
        steps.append(line)
        if len(steps) > 100:
            steps.popleft()
        y.append(np.average(steps))
        y1.append(line)
        x.append(i)

    plt.plot(x, y1, alpha=0.5, label="raw_data")
    plt.plot(x, y, color="red", label="Avg(100)")

    plt.xlabel('Training iteration')
    plt.ylabel('Learning rate decay')
    plt.legend()
    plt.savefig(folder_url + "/Charts/lr_decay_chart.png", transparent=True)


# create_charts()
