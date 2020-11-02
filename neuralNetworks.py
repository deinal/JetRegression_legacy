import tensorflow as tf
import tensorflow.keras as keras

def create_pfcand_network(nGlobalVariables, nChgPfCandidates, nChgPfVariables, nNeuPfCandidates, nNeuPfVariables,  nPhoPfCandidates, nPhoPfVariables):
    '''
    Creates a neural network similar to the original DeepJet in architecture
    :param nGlobalVariables: number of global training variables
    :param nChgPfCandidates: max number of charged pf candidates used
    :param nChgPfVariables: number of variables per chg. pf candidate
    :param nNeuPfCandidates: max number of neu. pf candidates
    :param nNeuPfVariables: number of variables per neu. pf candidate
    :param nPhoPfCandidates:  max number of pho. pf candidates
    :param nPhoPfVariables: number of variables per pho. pf candidate
    :return: neural network ready for training
    '''
    _regularization = tf.keras.regularizers.l2(5e-5)
    _initializer = "lecun_normal"
    _drop = 0.1
    _lstmUnits = 10

    inputChg = keras.layers.Input(shape=(nChgPfCandidates, nChgPfVariables), name="chg_inp")
    inputNeu = keras.layers.Input(shape=(nNeuPfCandidates, nNeuPfVariables), name="neu_inp")
    inputPho = keras.layers.Input(shape=(nPhoPfCandidates, nPhoPfVariables), name="pho_inp")
    inputGlobal = keras.layers.Input(nGlobalVariables, name="glo_inp")

    chg = keras.layers.Conv1D(32, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(inputChg)
    chg = keras.layers.BatchNormalization()(chg)
    chg_1 = keras.layers.Dropout(_drop)(chg)
    chg = keras.layers.Conv1D(16, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(chg_1)
    chg = keras.layers.BatchNormalization()(chg)
    chg = keras.layers.Dropout(_drop)(chg)
    chg = keras.layers.Conv1D(8, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(chg)
    chg = keras.layers.BatchNormalization()(chg)
    chg = keras.layers.Dropout(_drop)(chg)
    chg = keras.layers.Conv1D(2, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(chg)
    chg = keras.layers.BatchNormalization()(chg)

    chg_lstm = keras.layers.LSTM(_lstmUnits, go_backwards=True, dropout=_drop)(chg)

    neu = keras.layers.Conv1D(16, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(inputNeu)
    neu = keras.layers.BatchNormalization()(neu)
    neu_1 = keras.layers.Dropout(_drop)(neu)
    neu = keras.layers.Conv1D(8, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(neu_1)
    neu = keras.layers.BatchNormalization()(neu)
    neu = keras.layers.Dropout(_drop)(neu)
    neu = keras.layers.Conv1D(4, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(neu)
    neu = keras.layers.BatchNormalization()(neu)
    neu = keras.layers.Dropout(_drop)(neu)
    neu = keras.layers.Conv1D(2, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(neu)
    neu = keras.layers.BatchNormalization()(neu)

    neu_lstm = keras.layers.LSTM(_lstmUnits, go_backwards=True, dropout=_drop)(neu)

    pho = keras.layers.Conv1D(16, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(inputPho)
    pho = keras.layers.BatchNormalization()(pho)
    pho_1 = keras.layers.Dropout(_drop)(pho)
    pho = keras.layers.Conv1D(8, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(pho_1)
    pho = keras.layers.BatchNormalization()(pho)
    pho = keras.layers.Dropout(_drop)(pho)
    pho = keras.layers.Conv1D(4, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(pho)
    pho = keras.layers.BatchNormalization()(pho)
    pho = keras.layers.Dropout(_drop)(pho)
    pho = keras.layers.Conv1D(2, kernel_size=1, activation='relu', activity_regularizer=_regularization,
                              kernel_initializer=_initializer)(pho)
    pho = keras.layers.BatchNormalization()(pho)

    pho_lstm = keras.layers.LSTM(_lstmUnits, go_backwards=True, dropout=_drop)(pho)

    concat = keras.layers.Concatenate()([inputGlobal, chg_lstm, neu_lstm, pho_lstm])


    dense = keras.layers.Dense(256, activation='relu', activity_regularizer=_regularization,
                               kernel_initializer=_initializer)(concat)
    dense = keras.layers.BatchNormalization()(dense)
    dense = keras.layers.Dropout(_drop)(dense)
    dense = keras.layers.Dense(128, activation='relu', activity_regularizer=_regularization,
                               kernel_initializer=_initializer)(dense)
    dense = keras.layers.BatchNormalization()(dense)
    dense = keras.layers.Dropout(_drop)(dense)
    dense = keras.layers.Dense(64, activation='relu', activity_regularizer=_regularization,
                               kernel_initializer=_initializer)(dense)
    dense = keras.layers.BatchNormalization()(dense)
    dense = keras.layers.Dropout(_drop)(dense)
    dense = keras.layers.Dense(32, activation='relu', activity_regularizer=_regularization,
                               kernel_initializer=_initializer)(dense)
    dense = keras.layers.BatchNormalization()(dense)
    dense = keras.layers.Dropout(_drop)(dense)

    output = keras.layers.Dense(1, activation='relu', activity_regularizer=_regularization,
                               kernel_initializer=_initializer,
                               bias_initializer=tf.keras.initializers.Constant(1.0))(dense)

    model = keras.Model([inputGlobal, inputChg, inputNeu, inputPho], output)
    loss_ = tf.keras.losses.Huber()
    model.compile(optimizer=keras.optimizers.Adam(lr=3e-4, amsgrad=True),
                  loss=loss_)

    print(model.summary())
    return model
