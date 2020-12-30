'''
Plotting related code
'''

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from matplotlib.transforms import ScaledTranslation


def getBinnedStatistics(variable, binningVariable, binning):
    indices = np.digitize(binningVariable, binning, right=True)
    indices[indices == 0] = 1
    indices = indices - 1

    binMean = -np.ones(len(binning))
    binStd = np.zeros(len(binning))

    for i in np.unique(indices):
        if (np.sum(indices == i) < 2):
            continue
        mean, std = norm.fit(variable[indices == i])
        binMean[i] = mean
        binStd[i] = std

    return binMean, binStd


def plotInclusiveResponse(dataframe, name):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6), constrained_layout=True)
    fig.suptitle("Inclusive response")
    trans1 = ax.transData + ScaledTranslation(-5 / 72, 0, fig.dpi_scale_trans)
    trans2 = ax.transData + ScaledTranslation(+5 / 72, 0, fig.dpi_scale_trans)

    colors = [sns.xkcd_rgb["cerulean"], sns.xkcd_rgb["rouge"]]
    gluons_l1l2l3 = dataframe.loc[dataframe.isPhysG == 1, "jetPt"] / dataframe.loc[dataframe.isPhysG == 1, "genJetPt"]
    uds_l1l2l3 = dataframe.loc[dataframe.isPhysG != 1, "jetPt"] / dataframe.loc[dataframe.isPhysG != 1, "genJetPt"]
    gluons_dnn = dataframe.loc[dataframe.isPhysG == 1, "response"]
    uds_dnn = dataframe.loc[dataframe.isPhysG != 1, "response"]

    mean_g_l1l2l3, std_g_l1l2l3 = norm.fit(gluons_l1l2l3)
    mean_uds_l1l2l3, std_uds_l1l2l3 = norm.fit(uds_l1l2l3)
    median_g_l1l2l3 = np.median(gluons_l1l2l3)
    median_uds_l1l2l3 = np.median(uds_l1l2l3)
    mean_g_dnn, std_g_dnn = norm.fit(gluons_dnn)
    mean_uds_dnn, std_uds_dnn = norm.fit(uds_dnn)
    median_g_dnn = np.median(gluons_dnn)
    median_uds_dnn = np.median(uds_dnn)

    iqr_g_l1l2l3 = np.subtract(*np.percentile(gluons_l1l2l3, [75, 25]))*0.1
    iqr_uds_l1l2l3 = np.subtract(*np.percentile(uds_l1l2l3, [75, 25]))*0.1
    iqr_g_dnn = np.subtract(*np.percentile(gluons_dnn, [75, 25]))*0.1
    iqr_uds_dnn = np.subtract(*np.percentile(uds_dnn, [75, 25]))*0.1

    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(0.975, 1.04)
    ax.text(-0.5, 1.041, "60 GeV < p$_T$ < 600 GeV, |$\eta$| < 2.5", fontsize=7)
    ax.set_ylabel("Median response")
    ax.set_xlabel("Jet class")
    plt.errorbar(x=['g', 'uds'], y=[median_g_l1l2l3, median_uds_l1l2l3], fmt='o', label="L1L2L3", color=colors[1], ls='none', ecolor='k', transform=trans1)
    plt.errorbar(x=['g', 'uds'], y=[median_g_dnn, median_uds_dnn], yerr=[iqr_g_dnn, iqr_uds_dnn], fmt='o', label="DNN", color=colors[0], ls='none', ecolor='k', transform=trans2)
    ax.plot([-1.0, 2.0], [1.0, 1.0], linestyle='--', linewidth=1.5, color='k')
    ax.legend()
    plt.savefig("./plots/InclusiveResponse.pdf".format(name))
    plt.clf()
    plt.close()


def plotResidual(dataframe, name):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12, 6))
    fig.suptitle("Mean response residuals w.r.t gen p$_{T}$")
    binning = np.linspace(60.0, 600.0, 55)
    binCenters = binning+(binning[1]-binning[0])/2.0
    colors = [sns.xkcd_rgb["cerulean"], sns.xkcd_rgb["rouge"]]
    binningVariable = dataframe.loc[:, "genJetPt"]
    variable_1, std_1 = getBinnedStatistics(dataframe[(dataframe.isPhysG == 1)].loc[:, "response"], binningVariable[(dataframe.isPhysG == 1)], binning)
    variable_2, std_2 = getBinnedStatistics(dataframe[(dataframe.isPhysG != 1)].loc[:, "response"], binningVariable[(dataframe.isPhysG != 1)], binning)
    binMean = variable_1 - variable_2
    binStd = np.sqrt(std_1 ** 2 + std_2 ** 2)
    up = np.add(binMean, binStd)
    down = np.add(binMean, -binStd)

    controlVariable_1, std_1 = getBinnedStatistics(dataframe.loc[dataframe.isPhysG == 1, "jetPt"]/dataframe.loc[dataframe.isPhysG == 1, "genJetPt"], binningVariable[dataframe.isPhysG == 1], binning)
    controlVariable_2, std_2 = getBinnedStatistics(dataframe.loc[dataframe.isPhysG != 1, "jetPt"]/dataframe.loc[dataframe.isPhysG != 1, "genJetPt"], binningVariable[dataframe.isPhysG != 1], binning)
    controlBinMean = controlVariable_1 - controlVariable_2
    controlBinStd = np.sqrt(std_1 ** 2 + std_2 ** 2)
    controlUp = np.add(controlBinMean, controlBinStd)
    controlDown = np.add(controlBinMean, -controlBinStd)

    nonzeroPoints = (binMean > -1)
    ax0.fill_between(binCenters[nonzeroPoints], up[nonzeroPoints], down[nonzeroPoints], alpha=0.6, color=colors[0])
    ax0.plot(binCenters[nonzeroPoints], up[nonzeroPoints], c=colors[0], linewidth=1.0, alpha=0.8)
    ax0.plot(binCenters[nonzeroPoints], down[nonzeroPoints], c=colors[0], linewidth=1.0, alpha=0.8)
    ax0.scatter(binCenters[nonzeroPoints], binMean[nonzeroPoints], c=colors[0], label="DNN", linewidth=1.0, edgecolors='k')

    # ax0.fill_between(binCenters[nonzeroPoints], controlUp[nonzeroPoints], controlDown[nonzeroPoints], alpha=0.6, color=colors[1])
    ax0.plot(binCenters[nonzeroPoints], controlUp[nonzeroPoints], c=colors[1], linewidth=1.5, linestyle='--', alpha=0.8)
    ax0.plot(binCenters[nonzeroPoints], controlDown[nonzeroPoints], c=colors[1], linewidth=1.5, linestyle='--', alpha=0.8)
    ax0.scatter(binCenters[nonzeroPoints], controlBinMean[nonzeroPoints], c=colors[1], label="L1L2L3", linewidth=1.0, edgecolors='k')

    ax0.plot([binCenters[0], binCenters[-1]], [0.0, 0.0], linestyle='--', linewidth=1.5, color='k')

    ax0.set_ylim(-0.2, 0.2)
    locs = np.linspace(-0.2, 0.2, 21)
    ax0.set_yticks(locs)
    ax0.set_xlim(binning[0], binning[-1])
    ax0.set_ylabel("R$_G$-R$_{UDS}$")
    ax1.set_xlabel("gen p$_{T}$")

    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles[::-1], labels[::-1], loc='upper right')

    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.grid(False, axis='x')
    ax0.plot([binning[0], binning[-1]], [1.0, 1.0], linestyle='--', linewidth=1.5, color='k')

    ax1.set_xlim(binning[0], binning[-1])
    ax1.hist(dataframe.loc[:, "genJetPt"], bins=binning, alpha=1.0, edgecolor="black", linewidth=1.0, color=colors[0])
    ax1.set_ylabel("Jets/bin")

    plt.tight_layout(pad=2.2)
    fig.align_ylabels((ax0, ax1))
    plt.savefig("./plots/MeanResidual.pdf".format(name))
    plt.clf()
    plt.close()


def plotCorrectedPtVsGenPt(dataframe, binningToUse, name):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(12, 6))
    fig.suptitle("Mean response with respect to genPt ({})".format(name))
    variableName = "response"
    binning = binningToUse
    binCenters = binning+(binning[1]-binning[0])/2.0
    colors = [sns.xkcd_rgb["cerulean"], sns.xkcd_rgb["rouge"]]

    variable = dataframe.loc[:, variableName]
    binningVariable = dataframe.loc[:, "genJetPt"]

    binMean, binStd = getBinnedStatistics(variable, binningVariable, binning)
    up = np.add(binMean, binStd)
    down = np.add(binMean, -binStd)

    controlBinMean, controlBinStd = getBinnedStatistics(dataframe.loc[:, "jetPt"]/dataframe.loc[:, "genJetPt"], binningVariable, binning)
    controlUp = np.add(controlBinMean, controlBinStd)
    controlDown = np.add(controlBinMean, -controlBinStd)

    nonzeroPoints = (binMean > -1)
    ax0.fill_between(binCenters[nonzeroPoints], up[nonzeroPoints], down[nonzeroPoints], alpha=0.6, color=colors[0])
    ax0.plot(binCenters[nonzeroPoints], up[nonzeroPoints], c=colors[0], linewidth=1.0, alpha=0.8)
    ax0.plot(binCenters[nonzeroPoints], down[nonzeroPoints], c=colors[0], linewidth=1.0, alpha=0.8)
    ax0.scatter(binCenters[nonzeroPoints], binMean[nonzeroPoints], c=colors[0], label="DNN", linewidth=1.0, edgecolors='k')

    # ax0.fill_between(binCenters[nonzeroPoints], controlUp[nonzeroPoints], controlDown[nonzeroPoints], alpha=0.6, color=colors[1])
    ax0.plot(binCenters[nonzeroPoints], controlUp[nonzeroPoints], c=colors[1], linewidth=1.5, linestyle='--', alpha=0.8)
    ax0.plot(binCenters[nonzeroPoints], controlDown[nonzeroPoints], c=colors[1], linewidth=1.5, linestyle='--', alpha=0.8)
    ax0.scatter(binCenters[nonzeroPoints], controlBinMean[nonzeroPoints], c=colors[1], label="L1L2L3", linewidth=1.0, edgecolors='k')

    ax0.plot([binCenters[0], binCenters[-1]], [1.0, 1.0], linestyle='--', linewidth=1.5, color='k')

    ax0.set_ylim(0.8, 1.2)
    locs = np.linspace(0.8, 1.2, 21)
    ax0.set_yticks(locs)
    ax0.set_xlim(binning[0], binning[-1])
    ax0.set_ylabel("Mean response")

    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles[::-1], labels[::-1], loc='upper right')

    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.grid(False, axis='x')

    ax1.set_xlim(binning[0], binning[-1])
    ax1.hist(dataframe.loc[:, "genJetPt"], bins=binning, alpha=1.0, edgecolor="black", linewidth=1.0, color=colors[0])
    ax1.set_ylabel("Jets/bin")

    plt.tight_layout(pad=2.2)
    fig.align_ylabels((ax0, ax1))
    plt.savefig("./plots/MeanResponse_{}.pdf".format(name))
    plt.clf()
    plt.close()


def plotLossHistory(history):
    colors = [sns.xkcd_rgb["gold"], sns.xkcd_rgb["grass"]]

    loss = history.history["loss"][1:]
    valLoss = history.history["val_loss"][1:]
    epochs = range(1, len(loss)+2)[1:]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label="Training loss", color=colors[0], linewidth=2.0)
    plt.plot(epochs, valLoss, label="Validation loss", color=colors[1], linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss function value")
    plt.yscale('log')
    fig.suptitle("Training and validation loss")
    plt.legend()
    plt.savefig("./plots/Losses.pdf")
    plt.clf()
    plt.close()
