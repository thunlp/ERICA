import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sns.set(font="Times New Roman")
sns.set_theme(style="darkgrid")

plt.figure(figsize=(16,4))

def plot(df, ax):
    a = sns.lineplot(x="size of pretraining data", y="test F1", hue="ratio of DocRED", style="ratio of DocRED", data=df, markers=True, dashes=False, ax = ax)
    a.tick_params(labelsize=22)
    a.legend_.remove()
    # a.set_xlabel('size of pretraining data',fontsize=22)
    # a.set_ylabel('test IgF1',fontsize=22)
    # legend = plt.legend(loc='right', bbox_to_anchor=(1, 0.4), prop={'size': 10}, title='ratio of DocRED', fontsize='xx-small')
    return a

    plt.rc('font',family='Times New Roman')

with PdfPages("rel.pdf") as pdf:
    ax = plt.subplot(1, 3, 1)
    df = pd.read_csv('t1.csv')
    a = plot(df, ax)
    a.set_xlabel('1% DocRED',fontsize=22)
    a.set_ylabel(' ',fontsize=22)


    ax = plt.subplot(1, 3, 2)
    df = pd.read_csv('t2.csv')
    a = plot(df, ax)
    a.set_xlabel('10% DocRED',fontsize=22)
    a.set_ylabel(' ',fontsize=22)

    ax = plt.subplot(1, 3, 3)
    df = pd.read_csv('t4.csv')
    a = plot(df, ax)
    a.set_xlabel('100% DocRED',fontsize=22)
    a.set_ylabel(' ',fontsize=22)


    # plt.show()
    pdf.savefig(bbox_inches='tight')
    plt.close()