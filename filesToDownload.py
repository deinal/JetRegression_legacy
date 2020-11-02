'''
This file simply contains the urls of the JetNTuple files stored in the opendata portal. If a local copy of
flattened versions of these files does not exist, the code downloads and reformats them during the first run.
See: http://opendata.cern.ch/record/12100 for guidance and how to cite.
'''

filelist = [
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_1.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_10.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_100.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_101.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_102.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_103.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_104.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_105.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_106.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_107.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_108.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_109.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_11.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_110.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_111.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_112.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_113.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_114.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_115.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_116.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_117.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_118.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_119.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_12.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_120.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_121.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_122.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_13.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_14.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_15.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_16.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_17.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_18.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_19.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_2.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_20.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_21.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_22.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_23.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_24.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_25.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_26.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_27.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_28.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_29.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_3.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_30.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_31.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_32.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_33.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_34.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_35.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_36.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_37.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_38.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_39.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_4.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_40.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_41.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_42.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_43.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_44.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_45.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_46.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_47.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_48.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_49.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_5.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_50.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_51.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_52.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_53.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_54.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_55.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_56.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_57.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_58.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_59.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_6.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_60.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_61.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_62.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_63.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_64.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_65.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_66.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_67.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_68.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_69.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_7.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_70.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_71.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_72.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_73.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_74.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_75.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_76.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_77.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_78.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_79.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_8.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_80.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_81.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_82.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_83.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_84.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_85.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_86.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_87.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_88.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_89.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_9.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_90.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_91.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_92.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_93.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_94.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_95.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_96.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_97.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_98.root",
"http://opendata.cern.ch/record/12100/files/assets/cms/datascience/JetNtupleProducerTool/JetNTuple_QCD_RunII_13TeV_MC/JetNtuple_RunIISummer16_13TeV_MC_99.root",
]