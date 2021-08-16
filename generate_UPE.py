import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def start(u_model_path, p_model_path, e_model_path):

    u_model = pd.read_excel(u_model_path).values[:, 1:]
    p_model = pd.read_excel(p_model_path).values[:, 1:]
    e_model = pd.read_excel(e_model_path).values[:, 1:]

    u_data = np.zeros([14, 15])
    p_data = np.zeros([14, 15])
    e_data = np.zeros([14, 15])

    #Utility Model
    for i in range(1, 15):
        for j in range(15):
            utility_loss = 1 - (u_model[i, j]/u_model[0, 0])
            u_data[i-1, j] = utility_loss

    #Privacy Model
    for i in range(1, 15):
        for j in range(15):
            privacy_loss = (p_model[i, j]/p_model[0, 0])
            p_data[i-1, j] = privacy_loss

    #Energy Model
    for i in range(0, 14):
        for j in range(15):
            energy_loss = e_model[i, j]
            e_data[i, j] = energy_loss

    return u_data, p_data, e_data


def plot_best_region(u_data, p_data, e_data):
    from matplotlib import cm
    import matplotlib as mpl

    u_data = u_data
    p_data = p_data
    e_data = e_data

    font_size = 16

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    s = ax.plot_surface(e_data, u_data, p_data, linewidth=1, cmap='viridis', label='Global AVG Blur configurations')
    s2 = ax.scatter(e_data, u_data, p_data, linewidth=0, color='blue', s=7)

    ax.set_xlabel('Energy Loss', fontsize=font_size, labelpad=14)
    plt.xticks([0.88, 0.91, 0.94, 0.97, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_ylabel('Utility Loss', fontsize=font_size, labelpad=15)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Privacy Loss', fontsize=font_size, labelpad=10,rotation=90)
    #plt.legend()

    # # Second subplot
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    #
    # s2 = ax.scatter(e_data, u_data, p_data,linewidth = 0, cmap=mpl.cm.Reds)
    # ax.set_title('Utility x Privacy x Energy (Scatter Plot)')
    #
    # ax.set_xlabel('Energy', fontsize=font_size, labelpad=10)
    # ax.set_ylabel('Utility', fontsize=font_size, labelpad=10)
    # ax.set_zlabel('Privacy', fontsize=font_size, labelpad=13)
    #
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    #
    # ax.set_title('Utility x Privacy x Energy (Wireframe Plot)')
    # s3 = ax.plot_wireframe(e_data, u_data, p_data, alpha=0.8, cmap='viridis')
    # s2 = ax.scatter(e_data, u_data, p_data, linewidth=0, cmap=mpl.cm.Reds)
    # ax.set_xlabel('Energy', fontsize=font_size, labelpad=10)
    # ax.set_ylabel('Utility', fontsize=font_size, labelpad=10)
    # ax.set_zlabel('Privacy', fontsize=font_size, labelpad=13)

    plt.show()
    pass


def plot_best_region_multiple_privatizers(models_list):
    from matplotlib import cm
    import matplotlib as mpl

    font_size = 14

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for matrix in models_list:

        u_data = matrix[1]
        p_data = matrix[2]
        e_data = matrix[3]

        s2 = ax.scatter(e_data, u_data, p_data, linewidth=0, color=matrix[4], label=matrix[0], s=9)
        s2 = ax.plot_wireframe(e_data, u_data, p_data, color=matrix[4], linewidth=1,alpha=0.7)

    ax.set_xlabel('Energy Loss', fontsize=font_size, labelpad=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_ylabel('Utility Loss', fontsize=font_size, labelpad=15)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Privacy Loss', fontsize=font_size, labelpad=8, rotation=90)
    plt.legend(fontsize=12, bbox_to_anchor=(1.1, 1.15))

    plt.show()
    pass


if __name__ == "__main__":

    matrix_list = []

    u_data, p_data, e_data = start('F:/results/f1_hd_med_results.xlsx',
                                   'F:/results/f1_fm_med_results.xlsx',
                                   'F:/results/f1_energy_med_results.xlsx')
    #plot_best_region(u_data, p_data, e_data)

    #matrix_list.append(['Median Blur Privatizer', u_data, p_data, e_data, 'green'])

    u_data, p_data, e_data = start('F:/results/f1_hd_avg_results.xlsx',
                                   'F:/results/f1_fm_avg_results.xlsx',
                                   'F:/results/f1_energy_avg_results.xlsx')

    matrix_list.append(['Average Blur Privatizer', u_data, p_data, e_data, 'red'])

    u_data, p_data, e_data = start('F:/results/f1_hd_fdet_results.xlsx',
                                   'F:/results/f1_fm_fdet_results.xlsx',
                                   'F:/results/f1_energy_fdet_results.xlsx')

    matrix_list.append(['Face Blur Privatizer', u_data, p_data, e_data, 'blue'])

    plot_best_region_multiple_privatizers(matrix_list)

    #plot_best_region(u_data, p_data, e_data)

    pass
