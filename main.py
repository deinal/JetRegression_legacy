#Sets the GPU to the least used one. Has to be imported first. See gpusetter.py
import gpusetter

from IOFunctions import get_data, format_pf_candidates_for_convolutions, create_directories
from preprocess import preprocess_dataframe, get_training_sample_weights
from neuralNetworks import create_pfcand_network
from plotting import plotLossHistory, plotCorrectedPtVsGenPt, plotResidual, plotInclusiveResponse
import numpy as np
import config

#Print all elements in numpy array
np.set_printoptions(threshold=np.inf)
#======================

#Initialize necessary directories
create_directories()
#======================

#Read data to memory (or download it if a local copy is missing or explicitly requested)
dataframe = get_data(remake_flat_frames=False)
#======================

#Set training target as the required correction and select the phase space of which jets to include
#NOTE: Already in generating the flat tuples, some level of selection is done in jets, see IOFunctions.py
dataframe.loc[:, "target"] = dataframe.loc[:, "genJetPt"]/dataframe.loc[:, "jetPt"]
dataframe = dataframe.loc[(dataframe.genJetPt > 30) & ((dataframe.target > 0.8) & (dataframe.target <= 1.3)), :]
#======================

#Variables used in training
trainingVariables = config.globalVariables+config.flattenedChgParticleVariables+config.flattenedNeuParticleVariables+config.flattenedPhoParticleVariables
#======================

#Split dataframe to training and test samples
testDataframe = dataframe.sample(n=100000)
dataframe = dataframe.drop(testDataframe.index)
dataframe = dataframe.sample(frac=1.0)
dataframe.reset_index(inplace=True, drop=True)
testDataframe.reset_index(inplace=True, drop=True)
#======================

#Compute training weights
sampleWeights = get_training_sample_weights(dataframe)
#======================

#Declare targets for training
targets = dataframe.loc[:, "target"]
testTargets = testDataframe.loc[:, "target"]
#======================

#Create a copy of the dataframes that will be scaled.
#This is a dumb way to preserve the original frames (useful for plotting)
#but still do the training with some variables being scaled.
#pT variables are log1p scaled to account for the huge differences in scales,
#and after that each variable is normalized.
scalable_glob = ["jetPt"]+[s for s in config.globalVariables if 'ring' in s]
scalable_chg = [s for s in config.flattenedChgParticleVariables if '_pT' in s]
scalable_neu = [s for s in config.flattenedNeuParticleVariables if '_pT' in s]
scalable_pho = [s for s in config.flattenedPhoParticleVariables if '_pT' in s]

scaled_testDataframe = testDataframe.copy()
scaled_testDataframe.loc[:, scalable_glob+scalable_chg+scalable_neu+scalable_pho] = scaled_testDataframe.loc[:, scalable_glob+scalable_chg+scalable_neu+scalable_pho].applymap(np.log1p)
scaled_testDataframe = preprocess_dataframe(testDataframe.copy(), trainingVariables)
scaled_dataframe = dataframe.copy()
scaled_dataframe.loc[:, scalable_glob+scalable_chg+scalable_neu+scalable_pho] = scaled_dataframe.loc[:, scalable_glob+scalable_chg+scalable_neu+scalable_pho].applymap(np.log1p)
scaled_dataframe = preprocess_dataframe(scaled_dataframe, trainingVariables)

scaled_dataframe.reset_index(inplace=True, drop=True)
testDataframe.reset_index(inplace=True, drop=True)
scaled_testDataframe.reset_index(inplace=True, drop=True)
#==================================

#Training loop. Using strategy allows for use of multiple GPUs to run the training if set so in gpusetter.py
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())if len(tf.config.list_logical_devices('GPU')) > 1 else tf.distribute.get_strategy()
with strategy.scope():

    #Reduce learning rate when nearing convergence
    reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=1e-8
    )
    # ===================
    #Early stop if the network stops improving
    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=7, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    # ===================

    #Fetch the network
    network = create_pfcand_network(len(config.globalVariables),
                                    config.nChgPfCandidates, config.nChgPfVariables,
                                    config.nNeuPfCandidates, config.nNeuPfVariables,
                                    config.nPhoPfCandidates, config.nPhoPfVariables)
    # ===================

    #Split into validation dataset that can be monitored during training and training dataset
    val_dataset = scaled_dataframe.sample(frac=0.15)
    val_targets = targets.iloc[val_dataset.index]
    scaled_dataframe = scaled_dataframe.drop(val_dataset.index)
    targets = targets.drop(val_targets.index)
    sampleWeights = sampleWeights[scaled_dataframe.index]
    sampleWeights = np.ones(scaled_dataframe.shape[0])
    sample_weights = tf.convert_to_tensor(sampleWeights, dtype=tf.float32)
    #===================

    #Convert pandas dataframes to tf datasets for training
    dataset = tf.data.Dataset.from_tensor_slices(({
        "chg_inp": format_pf_candidates_for_convolutions(scaled_dataframe.loc[:, config.flattenedChgParticleVariables], config.nChgPfVariables),
        "neu_inp": format_pf_candidates_for_convolutions(scaled_dataframe.loc[:, config.flattenedNeuParticleVariables], config.nNeuPfVariables),
        "pho_inp": format_pf_candidates_for_convolutions(scaled_dataframe.loc[:, config.flattenedPhoParticleVariables], config.nPhoPfVariables),
        "glo_inp": scaled_dataframe.loc[:, config.globalVariables]
    }, targets, sample_weights))
    val_dataset = tf.data.Dataset.from_tensor_slices(({
        "chg_inp": format_pf_candidates_for_convolutions(val_dataset.loc[:, config.flattenedChgParticleVariables], config.nChgPfVariables),
        "neu_inp": format_pf_candidates_for_convolutions(val_dataset.loc[:, config.flattenedNeuParticleVariables], config.nNeuPfVariables),
        "pho_inp": format_pf_candidates_for_convolutions(val_dataset.loc[:, config.flattenedPhoParticleVariables], config.nPhoPfVariables),
        "glo_inp": val_dataset.loc[:, config.globalVariables]
    }, val_targets))
    # ===================

    #Shuffle, batch and prefetch datasets
    dataset = dataset.shuffle(10*config.batch_size)\
        .batch(config.batch_size*len(tf.config.list_logical_devices('GPU')), drop_remainder=True)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(config.batch_size)
    # ===================

    #Run training
    history = network.fit(
            dataset,
            validation_data=val_dataset,
            epochs=5,
            callbacks=[reduceLROnPlateau, earlyStopping],
            )
    # ===================
# ===================

#Predictions for test dataset corrections
predictions = network.predict([scaled_testDataframe.loc[:, config.globalVariables], format_pf_candidates_for_convolutions(scaled_testDataframe.loc[:, config.flattenedChgParticleVariables], config.nChgPfVariables), format_pf_candidates_for_convolutions(scaled_testDataframe.loc[:, config.flattenedNeuParticleVariables], config.nNeuPfVariables), format_pf_candidates_for_convolutions(scaled_testDataframe.loc[:, config.flattenedPhoParticleVariables], config.nPhoPfVariables)],
                              batch_size=128,
                              use_multiprocessing=True, workers=12
                              )
# ===================

testDataframe.loc[:, "predicted"] = predictions
testDataframe.loc[:, "correctedPt"] = testDataframe.loc[:, "predicted"]*testDataframe.loc[:, "jetPt"]
testDataframe.loc[:, "response"] = testDataframe.loc[:, "correctedPt"]/testDataframe.loc[:, "genJetPt"]

#Plotting
binningToUse = np.linspace(30.0, 600.0, 54)
plotResidual(testDataframe, "Residual")
plotInclusiveResponse(testDataframe, "Inclusive")
plotCorrectedPtVsGenPt(testDataframe.loc[(testDataframe.isPhysUDS == 1), :], binningToUse, "UDS")
plotCorrectedPtVsGenPt(testDataframe.loc[(testDataframe.isPhysG == 1), :], binningToUse, "G")
plotLossHistory(history)
# ===================

#Store model for possible later use
network.save('./saved_model')
# ===================