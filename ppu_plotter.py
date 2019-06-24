from mpl_toolkits.mplot3d import axes3d
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def general_plot(energy_traces, fd_f1_measure, fd_fpr, fd_precision, fd_recall, fd_truth_recall, hd_f1_measure, hd_fpr, hd_precision, hd_recall,
                 x_holder, x_label):

    fig, axis = plt.subplots(2, 2)

    # Face Recall x Precision
    axis[0, 0].set_title('Face ID - Recall x Precision')
    axis[0, 0].plot(x_holder, fd_precision, 'r-')
    axis[0, 0].plot(x_holder, fd_recall, 'y-')
    axis[0, 0].plot(x_holder, fd_truth_recall, 'b-')
    axis[0, 0].plot(x_holder, fd_f1_measure, 'g-')
    twin_ax00 = axis[0, 0].twinx()
    twin_ax00.plot(x_holder, energy_traces, '--', color='lightgray')
    twin_ax00.set(ylabel='Energy (Joules)')
    r_leg00 = mlines.Line2D([], [], color='red', markersize=1, label='Precision')
    p_leg00 = mlines.Line2D([], [], color='yellow', markersize=1, label='Recall')
    tr_leg00 = mlines.Line2D([], [], color='blue', markersize=1, label='True Recall')
    f1_leg00 = mlines.Line2D([], [], color='green', markersize=1, label='F1')
    axis[0, 0].legend(handles=[r_leg00, p_leg00, f1_leg00, tr_leg00], loc=3)
    axis[0, 0].set(xlabel=x_label)
    axis[0, 0].set_ylim(0, 1)

    # Face ROC
    axis[1, 0].set_title('Face ID - RoC')
    axis[1, 0].plot(x_holder, fd_recall, 'b.')
    axis[1, 0].plot(x_holder, fd_truth_recall, 'y.')
    axis[1, 0].plot(x_holder, fd_fpr, 'r.')
    twin_ax10 = axis[1, 0].twinx()
    twin_ax10.plot(x_holder, energy_traces, '--', color='lightgray')
    twin_ax10.set(ylabel='Energy (Joules)')
    r_leg10 = mlines.Line2D([], [], color='red', markersize=1, label='TPR')
    p_leg10 = mlines.Line2D([], [], color='blue', markersize=1, label='FPR')
    axis[1, 0].legend(handles=[r_leg10, p_leg10], loc=2)
    axis[1, 0].set(xlabel=x_label)
    axis[1, 0].set_ylim(0, 1)

    # Human Detection Recall x Precision
    axis[0, 1].set_title('Human Detection - Recall x Precision')
    axis[0, 1].plot(x_holder, hd_precision, 'r.')
    axis[0, 1].plot(x_holder, hd_recall, 'b.')
    axis[0, 1].plot(x_holder, hd_f1_measure, 'g.')
    r_leg01 = mlines.Line2D([], [], color='red', markersize=1, label='Precision')
    p_leg01 = mlines.Line2D([], [], color='blue', markersize=1, label='Recall')
    f1_leg01 = mlines.Line2D([], [], color='green', markersize=1, label='F1')
    twin_ax01 = axis[0, 1].twinx()
    twin_ax01.plot(x_holder, energy_traces, '--', color='lightgray')
    twin_ax01.set(ylabel='Energy (Joules)')
    axis[0, 1].legend(handles=[r_leg01, p_leg01, f1_leg01], loc=3)
    axis[0, 1].set(xlabel=x_label)
    axis[0, 1].set_ylim(0, 1)

    # Human detection ROC
    axis[1, 1].set_title('Human Detection - RoC')
    axis[1, 1].plot(x_holder, hd_recall, 'b.')
    axis[1, 1].plot(x_holder, hd_fpr, 'r.')
    r_leg11 = mlines.Line2D([], [], color='red', markersize=1, label='FPR')
    p_leg11 = mlines.Line2D([], [], color='blue', markersize=1, label='TPR')
    twin_ax11 = axis[1, 1].twinx()
    twin_ax11.plot(x_holder, energy_traces, '--', color='lightgray')
    twin_ax11.set(ylabel='Energy (Joules)')
    axis[1, 1].legend(handles=[r_leg11, p_leg11], loc=2)
    axis[1, 1].set(xlabel=x_label)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axis[1, 1].set_ylim(0, 1)

    plt.show()


def plot_face_embeddings_data(df_data, energy_traces, fd_embeddings_distance, fd_embeddings_std, x_holder, x_label,
                              y_cut):
    fig, axis = plt.subplots(2, 1)
    # Face embeddings
    axis[0].set_title('Face Embeddings Distance')
    axis[0].errorbar(x_holder, fd_embeddings_distance, yerr=fd_embeddings_std, color='gray', fmt='.k',
                     ecolor='lightgray', elinewidth=1)
    axis[0].plot(x_holder, y_cut, '-', color='red')
    twin_ax0 = axis[0].twinx()
    twin_ax0.plot(x_holder, energy_traces, '-', color='gray')
    twin_ax0.set(ylabel='Energy (Joules)')
    axis[0].set(xlabel=x_label, ylabel='Embeddings distance')
    axis[0].set_ylim(0.3, 1.5)
    axis[1].set_title('Face Embeddings Distance - Minimum')
    axis[1].plot(x_holder, df_data[:, 9], 'b.')
    axis[1].plot(x_holder, y_cut, 'r-')
    axis[1].set(xlabel=x_label, ylabel='Embeddings distance')
    axis[1].set_ylim(0, 1.5)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def raw_attack_rate_metric(af_us_from_user_s_rate, as_uf_rate, energy_traces, us_af_from_a_fails_rate, x_holder, x_label):
    # Attack Success rate
    fig, axis = plt.subplots(2, 1)
    # Truth ROC
    axis[0].set_title('User & Adversary - Attack/Success rate')
    axis[0].plot(x_holder, af_us_from_user_s_rate, 'g-')
    axis[0].plot(x_holder, us_af_from_a_fails_rate, 'y-')
    axis[0].plot(x_holder, as_uf_rate, 'r-')
    g1_leg10 = mlines.Line2D([], [], color='green', markersize=1, label='AF US FUS')
    g2_leg10 = mlines.Line2D([], [], color='yellow', markersize=1, label='AF US FAF')
    r1_leg10 = mlines.Line2D([], [], color='red', markersize=1, label='AS UF FAS')
    twin_ax0 = axis[0].twinx()
    twin_ax0.plot(x_holder, energy_traces, '-', color='gray')
    twin_ax0.set(ylabel='Energy (Joules)')
    axis[0].legend(handles=[g1_leg10, g2_leg10, r1_leg10], loc=2)
    axis[0].set(xlabel=x_label)
    axis[1].set_ylim(0, 1)
    plt.show()


def roc_plot(energy_traces, fd_fpr, fd_recall, hd_fpr, hd_recall, x_holder, x_label):

    fig, axis = plt.subplots(2, 1)

    # Truth ROC
    axis[0].set_title('User & Adversary - Truth RoC')
    axis[0].plot(fd_fpr, fd_recall, 'r.')
    axis[0].plot(hd_fpr, hd_recall, 'b.')
    r_leg10 = mlines.Line2D([], [], color='red', markersize=1, label='Adv RoC')
    p_leg10 = mlines.Line2D([], [], color='blue', markersize=1, label='User RoC')
    axis[0].legend(handles=[r_leg10, p_leg10], loc=2)
    axis[0].set(xlabel='False Positive Rate (1 - Specificity)', ylabel='True Positive Rate (Sensitivity)')
    axis[0].set_ylim(0, 1)
    axis[0].set_xlim(0, 1)

    # Human detection ROC
    axis[1].set_title('User & Adversary - RoC')
    axis[1].plot(x_holder, hd_recall, 'b-')
    axis[1].plot(x_holder, hd_fpr, 'r-')
    axis[1].plot(x_holder, fd_recall, 'g-')
    axis[1].plot(x_holder, fd_fpr, 'y-')
    r_leg11 = mlines.Line2D([], [], color='red', markersize=1, label='User FPR')
    b_leg11 = mlines.Line2D([], [], color='blue', markersize=1, label='User TPR')
    g_leg11 = mlines.Line2D([], [], color='green', markersize=1, label='Adv TPR')
    y_leg11 = mlines.Line2D([], [], color='yellow', markersize=1, label='Adv FPR')
    twin_ax11 = axis[1].twinx()
    twin_ax11.plot(x_holder, energy_traces, '--', color='lightgray')
    twin_ax11.set(ylabel='Energy (Joules)')
    axis[1].legend(handles=[r_leg11, b_leg11, g_leg11, y_leg11], loc=2)
    axis[1].set(xlabel=x_label)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axis[1].set_ylim(0, 1)

    plt.show()
