'''
Central storage for lists of variables and numbers called in different parts of the code. Determinese which variables
are used and how many pfCandidates at maximum are taken into account per jet
'''


batch_size = 512
globalVariables = ["jetPt", "jetEta", "QG_ptD", "QG_axis2", "QG_mult", 'alpha']
pfVariables = ['PF_pT', 'PF_dR', 'PF_dPhi']
chgParticleVariables = ["jetPF_chg_pT", "jetPF_chg_dR", "jetPF_chg_dPhi"]
neuParticleVariables = ["jetPF_neu_pT", "jetPF_neu_dR", "jetPF_neu_dPhi"]
phoParticleVariables = ["jetPF_pho_pT", "jetPF_pho_dR", "jetPF_pho_dPhi"]
truthVariables = ["genJetPt", "isPhysUDS", "isPhysG"]
nChgPfCandidates = 20
nNeuPfCandidates =10
nPhoPfCandidates =10
nChgPfVariables = len(chgParticleVariables)
nNeuPfVariables = len(neuParticleVariables)
nPhoPfVariables = len(phoParticleVariables)

flattenedChgParticleVariables = []
flattenedNeuParticleVariables = []
flattenedPhoParticleVariables = []

flattenedChgParticleVariables = [x + "_" + str(index) for index in range(nChgPfCandidates) for x in
                                 chgParticleVariables]
flattenedNeuParticleVariables = [x + "_" + str(index) for index in range(nNeuPfCandidates) for x in
                                 neuParticleVariables]
flattenedPhoParticleVariables = [x + "_" + str(index) for index in range(nPhoPfCandidates) for x in
                                 phoParticleVariables]




