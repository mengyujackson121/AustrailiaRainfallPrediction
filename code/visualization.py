import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pie_chart(data, title = "Non-Title"):
    labels = []
    fracs = []
    for label, frac in data.items():
        labels.append(label)
        fracs.append(frac)
    fig, axs = plt.subplots(1, 1)

    # Shift the second slice using explode
    axs.pie(fracs, labels=labels, autopct='%.2f%%',shadow=True,
                  explode=(0.05,) * len(labels))
    plt.title(title, fontsize=16, weight = 'bold');

    plt.show()

def bar_chart(data, xlable, ylabel, title):
    labels = []
    fracs = []
    for label, frac in data.items():
        labels.append(label)
        fracs.append(frac)

    x_pos = [i for i, _ in enumerate(labels)]
    plt.bar(x_pos, fracs, color='red')
    plt.xlabel(xlable)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(x_pos, labels)
