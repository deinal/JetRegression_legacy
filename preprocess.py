'''
Functions used for preprocessing the dataframes for training
'''
import config
import numpy as np

def preprocess_dataframe(dataframe, trainingVariables):
    '''
    Scale all trainingVariables to be in range [-1.0, 1.0]
    :param dataframe:
    :param trainingVariables: variables used as training inputs to the neural network
    :return: dataframe with variables scaled
    '''
    dataframe.loc[:, trainingVariables] = 2.0*(dataframe.loc[:, trainingVariables] - dataframe.loc[:, trainingVariables].min()) / ( dataframe.loc[:, trainingVariables].max() - dataframe.loc[:, trainingVariables].min()) - 1.0
    dataframe.fillna(0.0, inplace=True)
    return dataframe

def get_energy_in_rings(dataframe):
    '''
    Constructs variables containing the amount of energy within an annulus between r, r+dr from jet center.
    This is done separately for charged, neutral and photon candidates.
    :param dataframe: flattened dataframe
    :return: flattened dataframe with additional variables.
    '''
    chgPt = ["jetPF_chg_pT_" + str(index) for index in range(config.nChgPfCandidates)]
    chgDr = ["jetPF_chg_dR_" + str(index) for index in range(config.nChgPfCandidates)]
    neuPt = ["jetPF_neu_pT_" + str(index) for index in range(config.nNeuPfCandidates)]
    neuDr = ["jetPF_neu_dR_" + str(index) for index in range(config.nNeuPfCandidates)]
    phoPt = ["jetPF_pho_pT_" + str(index) for index in range(config.nPhoPfCandidates)]
    phoDr = ["jetPF_pho_dR_" + str(index) for index in range(config.nPhoPfCandidates)]

    ind=0
    for threshold in [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.7)]:

        bitmask = dataframe.loc[:, chgDr].applymap(lambda x: (x > threshold[0] and x < threshold[1]))
        dataframe.loc[:, "ringChg_{}".format(ind)] = np.sum(np.multiply(bitmask, dataframe.loc[:, chgPt]), axis=1)

        bitmask = dataframe.loc[:, neuDr].applymap(lambda x: (x > threshold[0] and x < threshold[1]))
        dataframe.loc[:, "ringNeu_{}".format(ind)] = np.sum(np.multiply(bitmask, dataframe.loc[:, neuPt]), axis=1)

        bitmask = dataframe.loc[:, phoDr].applymap(lambda x: (x > threshold[0] and x < threshold[1]))
        dataframe.loc[:, "ringPho_{}".format(ind)] = np.sum(np.multiply(bitmask, dataframe.loc[:, phoPt]), axis=1)

        ind = ind+1

    return dataframe

def get_training_sample_weights(dataframe):
    '''
    Calculates per sample weights for training samples in dataframe, so that
    the distribution is wieghted flat w.r.t. binned responses of the jets. This
    tries to enforce the underrepresented samples with a response deviating
    significantly from 1.0 to affect the training more.
    :param dataframe:
    :return: weights as np.array(size=num_training_events)
    '''
    nEntries = dataframe.shape[0]
    responseBinning = np.linspace(0.8, 1.2, 41)
    trueResponse = np.clip(dataframe.loc[:, "genJetPt"]/dataframe.loc[:, "jetPt"], responseBinning[0], responseBinning[-1]-1e-5)

    digitizedResp = np.digitize(trueResponse, bins=responseBinning, right=True)
    digitizedResp[digitizedResp == 0] = 1
    digitizedResp = digitizedResp - 1

    binnedResp = np.ones(len(responseBinning))
    for i in np.unique(digitizedResp):
        binnedResp[i] = np.sum(digitizedResp == i)

    binMultipliers = np.divide(nEntries/len(np.unique(digitizedResp)), binnedResp)

    sampleWeights = binMultipliers[digitizedResp]

    return sampleWeights

# def find_max_pts(row):
#     row.loc["maxPtParticles"] = (-row.loc['PF_pT']).argsort()[:3]
#     return row
#
# def rotate_and_flip(row):
#     max_pt_indices = row.loc["maxPtParticles"]
#
#     dEta_vec = row.loc["PF_dEta"]
#     dEta_vec = dEta_vec - dEta_vec[max_pt_indices[1]]
#     dPhi_vec = row.loc["PF_dPhi"]
#     dPhi_vec = dPhi_vec - dPhi_vec[max_pt_indices[1]]
#     flip = dPhi_vec[max_pt_indices[2]] < 0.0
#     if flip:
#         dPhi_vec = -dPhi_vec
#     else:
#         dPhi_vec = dPhi_vec
#
#     row.loc["PF_dPhi"] = dPhi_vec
#     row.loc["PF_dEta"] = dEta_vec
#
#     return row
#
#
# def get_variables_for_jet_image(dataframe):
#     dataframe.loc[:, "maxPtParticles"] = dataframe.apply(find_max_pts, axis=1)
#     dataframe.loc[:, "PF_dPhi"] = dataframe.apply(rotate_and_flip, axis=1)
#     return dataframe.loc[:, ["genJetPt", "isPhysUDS", "isPhysG", "jetPt", "QG_ptD", "QG_axis2", "QG_mult", "PF_pT", "PF_dEta", "PF_dPhi"]]

def normalize(array):
    lmin = float(array.min())
    lmax = float(array.max())
    return (array-lmin)/(lmax-lmin)

def equalize(array):
    hist = np.histogram(array, bins=np.arange(257))[0]
    H = np.cumsum(hist)/float(np.sum(hist))
    e = np.floor(H[array.flatten().astype('int')])
    return e.reshape(array.shape)

# def readJetImages(path, start, end):
#     genPts = []
#     recoPts = []
#     isG = []
#     filepaths = glob.glob(path+"/*.npy")[start:end]
#     images = np.empty((len(filepaths), 30, 30))
#     for i, path in enumerate(filepaths):
#         numbers = re.findall(r'\d+\.\d+', path)
#         tag = re.findall(r'\_(.*?)\.', path)[-1]
#         if tag == 'G':
#             isG.append(True)
#         elif tag == 'UDS':
#             isG.append(False)
#         else:
#             os.remove(path)
#             continue
#         genPts.append(float(numbers[0]))
#         recoPts.append(float(numbers[1]))
#         images[i] = np.load(path)
#         images[i] = normalize(images[i])
#         if(np.max(images[i])==0.0):
#             os.remove(path)
#             continue
#
#     images = np.expand_dims(images, -1)
#     return np.array(genPts), np.array(recoPts), np.array(isG), images

# def create_jet_images(id, dataframe, path):
#     w = 30+1
#     h = 30+1
#     gridx = np.linspace(-1.0, 1.0, w)
#     gridy = np.linspace(-1.0, 1.0, h)
#     for i in range(dataframe.shape[0]):
#         row = dataframe.loc[i, :]
#         if (i+1)%1000==0:
#             print("Process {}: {} out of {}".format(id, (i+1), dataframe.shape[0]))
#         x = row.loc['PF_dPhi']
#         y = row.loc['PF_dEta']
#         pt = row.loc['PF_pT']
#         parton = 'G' if row.loc['isPhysG'] else 'UDS'
#         genPt = row.loc['genJetPt']
#         jetPt = row.loc['jetPt']
#         array, _, _, _ = plt.hist2d(x, y, bins=[gridx, gridy], weights=pt)
#         np.save(path+"/jetImage_{}_{}_{}.npy".format(round(genPt, 3), round(jetPt, 3), parton), array)


###
# def getTargets(dataframe):
#     responseBinning = np.linspace(0.8, 1.2, 21)
#
#     trueResponse = np.clip(dataframe.loc[:, "genJetPt"]/dataframe.loc[:, "jetPt"], responseBinning[0], responseBinning[-1]-1e-5)
#
#     digitizedResp = np.digitize(trueResponse, bins=responseBinning, right=True)
#     digitizedResp[digitizedResp == 0] = 1
#     digitizedResp = digitizedResp - 1
#     oneHot = tf.keras.utils.to_categorical(digitizedResp, len(responseBinning))
#
#     dictToConvertDigitized = {x: round(responseBinning[x], 2) for x in range(len(responseBinning))}
#
#     return oneHot, dictToConvertDigitized
#
