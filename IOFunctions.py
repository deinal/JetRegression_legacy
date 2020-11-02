
import glob
from uproot import open
import pandas as pd
import numpy as np
import os
import shutil

from filesToDownload import filelist
import urllib.request as request
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from preprocess import get_energy_in_rings

import config

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

flatFrameDir = "./data/flatFrames"

def get_data(remake_flat_frames=False):
    '''
    Fetches the dataframe either from open data storage or local copy if one is found
    (and remaking of flat frames is not explicitly requested)
    :param remake_flat_frames: if True, reload the files from open data portal and recreate local copies
    :return: dataframe with flattened jets
    '''
    folder_exists = os.path.exists(flatFrameDir)
    if not folder_exists or remake_flat_frames:
        if folder_exists:
            shutil.rmtree(flatFrameDir)
        os.makedirs(flatFrameDir)
        dataframe = download_and_flatten_dataframes()
    else:
        dataframe = read_flat_frames(glob.glob(f'{flatFrameDir}/*.pkl'))

    #silly way to add ring variables to global variables
    config.globalVariables = config.globalVariables+[s for s in dataframe.columns if 'ring' in s]
    return dataframe

def download_and_flatten_dataframes():
    '''
    If no local copy of the dataframes exist already, this function will initiate download from the
    open data repository, select the necessary variables and format the data into a flat format that
    can be used in the training code. This has a large overhead in running time, so the flattening
    should not be done too often. It takes O(1 hour) to run, but realistically one only needs to do
    this step when first running the code, or if there are some fundamental changes in which variables
    to include from the JetNTuple.

    :return: dataframe of the flat jets
    '''
    if os.path.exists("./tmp"):
        shutil.rmtree("./tmp")
    os.mkdir("./tmp")
    with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
        results = list(executor.map(process_url, tuple(zip(range(1, len(filelist) + 1), filelist))))
    dataframe = results[0].append(results[1:])
    shutil.rmtree("./tmp")
    return dataframe

def process_url(address):
    '''
    Downloads the file from the address, preselects the jets of interest with the selection conditions
    and formats them into a flat dataframe (so that the pfCandidate variables are in individual columns
    instead of being stored in vectors
    :param address: url to file storage, see filesToDownload.py
    :return: flattened dataframe where each row is a jet with scalar variable values in columns
    '''
    filename = address[1].rsplit('/', 1)[-1]
    print(f'Downloading: {filename}')
    savepath = './tmp/' + filename
    request.urlretrieve(address[1], savepath)
    tree = open(savepath)['AK4jets/jetTree'].arrays(namedecode='utf-8')
    dataframe = pd.DataFrame(tree, columns=tree.keys())
    isCorrectParton = (
                np.array(dataframe.loc[:, "isPhysUDS"].values == 1) | np.array(dataframe.loc[:, "isPhysG"].values == 1))
    nPfCandidates = dataframe.loc[:, "PF_pT"].values
    nPfCandidates = [len(candidates) >= 3 for candidates in nPfCandidates]
    conditions = ((dataframe.genJetPt > 30) & (dataframe.genJetPt < 1000) & (np.abs(dataframe.genJetEta) < 2.5) & isCorrectParton & nPfCandidates)
    dataframe = dataframe.loc[conditions, :]
    dataframe.reset_index(inplace=True, drop=True)
    dataframe = create_flat_frame(dataframe, address[0])
    os.remove(savepath)
    return dataframe

def read_flat_frames(filePaths):
    '''
    Multi-cpu read method to load data frames to memory
    :param filePaths:
    :return: flattened dataframe
    '''
    with ProcessPoolExecutor(max_workers=(cpu_count()-1)) as executor:
        results = list(executor.map(pd.read_pickle, filePaths))
    dataframe = results[0].append(results[1:])
    return dataframe

def create_flat_frame(dataframe, index):
    """
    Formats and saves locally a flattened dataframe
    :param dataframe: jetNtuple dataframe
    :param index: index used for storing the frame
    :return: flattened dataframe
    """
    globalFrame = dataframe.loc[:, config.truthVariables + config.globalVariables]
    dataframe = separate_into_charged_neutral_photon(dataframe)

    chgParticleFrame = flatten_particle_data_frame(dataframe.loc[:, config.chgParticleVariables], config.nChgPfCandidates)
    neuParticleFrame = flatten_particle_data_frame(dataframe.loc[:, config.neuParticleVariables], config.nNeuPfCandidates)
    phoParticleFrame = flatten_particle_data_frame(dataframe.loc[:, config.phoParticleVariables], config.nPhoPfCandidates)
    dataframe = pd.concat([globalFrame, chgParticleFrame, neuParticleFrame, phoParticleFrame], axis=1)
    dataframe = get_energy_in_rings(dataframe)
    dataframe.to_pickle(f"{flatFrameDir}/QCDjetsWithPU_{index}.pkl")
    return dataframe


def separate_into_charged_neutral_photon(dataframe):
    '''
    Takes a dataframe where pfCandidate variables are stored as vectors and
    splits them into columns of scalar values.
    :param dataframe: unflattened jetNtuple dataframe
    :return: flattened dataframe where pfCandidates have been split into columns
    '''
    #PF candidate IDs for charged/neutral hadrons and photons
    charged = [-211, -11, 11, 211]
    neutral = [310, 130, 111]
    photon = [22]
    chg_pts = []
    chg_drs = []
    chg_phis = []
    neu_pts = []
    neu_drs = []
    neu_phis = []
    pho_pts = []
    pho_drs = []
    pho_phis = []

    for index in range(dataframe.shape[0]):
        chargedPFs = np.array([x in charged for x in dataframe.loc[index, "PF_id"]])
        neutralPFs = np.array([x in neutral for x in dataframe.loc[index, "PF_id"]])
        photonPFs = np.array([x in photon for x in dataframe.loc[index, "PF_id"]])
        order_chg = np.flip(np.argsort(dataframe.loc[index, 'PF_pT'][chargedPFs]))
        order_neu = np.flip(np.argsort(dataframe.loc[index, 'PF_pT'][neutralPFs]))
        order_pho = np.flip(np.argsort(dataframe.loc[index, 'PF_pT'][photonPFs]))
        chg_pts.append(dataframe.loc[index, 'PF_pT'][chargedPFs].take(order_chg))
        chg_drs.append(dataframe.loc[index, 'PF_dR'][chargedPFs].take(order_chg))
        chg_phis.append(dataframe.loc[index, 'PF_dPhi'][chargedPFs].take(order_chg))
        neu_pts.append(dataframe.loc[index, 'PF_pT'][neutralPFs].take(order_neu))
        neu_drs.append(dataframe.loc[index, 'PF_dR'][neutralPFs].take(order_neu))
        neu_phis.append(dataframe.loc[index, 'PF_dPhi'][neutralPFs].take(order_neu))
        pho_pts.append(dataframe.loc[index, 'PF_pT'][photonPFs].take(order_pho))
        pho_drs.append(dataframe.loc[index, 'PF_dR'][photonPFs].take(order_pho))
        pho_phis.append(dataframe.loc[index, 'PF_dPhi'][photonPFs].take(order_pho))

    dataframe.loc[:, f"jetPF_chg_pT"] = np.array(chg_pts)
    dataframe.loc[:, f"jetPF_chg_dR"] = np.array(chg_drs)
    dataframe.loc[:, f"jetPF_chg_dPhi"] = np.array(chg_phis)
    dataframe.loc[:, f"jetPF_neu_pT"] = np.array(neu_pts)
    dataframe.loc[:, f"jetPF_neu_dR"] = np.array(neu_drs)
    dataframe.loc[:, f"jetPF_neu_dPhi"] = np.array(neu_phis)
    dataframe.loc[:, f"jetPF_pho_pT"] = np.array(pho_pts)
    dataframe.loc[:, f"jetPF_pho_dR"] = np.array(pho_drs)
    dataframe.loc[:, f"jetPF_pho_dPhi"] = np.array(pho_phis)
    return dataframe

def flatten_particle_data_frame(dataframe, nPfCandidates):
    columnLabels = [x+"_"+str(index) for x in dataframe.columns.values for index in range(nPfCandidates)]
    flatFrame = pd.concat([dataframe.loc[:, column].apply(lambda x: pd.Series(x).iloc[:nPfCandidates]) for column in dataframe.columns], axis=1, ignore_index=True)
    flatFrame.columns = columnLabels

    #Reshuffle labels for convolutions
    desiredColumnOrder = [x+"_"+str(index) for index in range(nPfCandidates) for x in dataframe.columns.values]
    flatFrame = flatFrame.loc[:, desiredColumnOrder]
    return flatFrame

def format_pf_candidates_for_convolutions(dataframe, nPfVariables):
    """
    Formats the pfCandidate variables into arrays of shape [nJets, nParticlesPerJet, nVariablesPerParticle]
    so that when a 1x1 convolution gets applied on the input, it convolutes different variables of the same
    particle together into a new representation for the particle. As the same convolution filters applied
    to every particle, the goal is to find a universally useful new representation for the input particles
    through 1x1 convolutions.
    :param dataframe: subframe containing the columns with the flattened pfCandidate variables of (type)
    :param nPfVariables: how many variables per particle are assigned for pfcandidates of (type)
    :return: reformatted subframe ready to be used as an input to the network
    """
    nParticles = int(dataframe.shape[1]/nPfVariables)
    array = dataframe.to_numpy()
    array = array.reshape(-1, nParticles, nPfVariables)
    return array

def create_directories():
    directories = ["plots", "logs"]

    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)