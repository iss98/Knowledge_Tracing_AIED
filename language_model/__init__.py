import os, json, sys, torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from .ModelInput_old import ModelInput
from .versionFind import versionFind
from .restoreProblem import restoreProblem

versionName = 'newVer_221215_20_26'
modelPath = os.path.join(os.getcwd(), 'language_model', 'reports', versionName)
sys.path.append(modelPath)

textTokenizerPath = os.path.join(modelPath, 'textTokenizer')
mathTokenizerPath = os.path.join(modelPath, 'mathTokenizer')
TEMLM_kwargsPath = os.path.join(modelPath, 'TEMLM_kwargs.json')
parameterPath = os.path.join(modelPath, versionName + '.pt')
datasetPath = os.path.join(os.getcwd(),'language_model', 'Datasets')

textTokenizer = AutoTokenizer.from_pretrained(textTokenizerPath)
mathTokenizer = AutoTokenizer.from_pretrained(mathTokenizerPath)

from TEMLM import TEMLM

with open(TEMLM_kwargsPath, 'r') as f:
    TEMLM_kwargs = json.load(f)

with open(os.path.join(datasetPath, 'train_probstat.json'), 'r', encoding= 'utf-8') as f:
    trainData = json.load(f)

lm = TEMLM(**TEMLM_kwargs)
lm.load_state_dict(torch.load(parameterPath))
#, map_location = torch.device('cpu')
trainInput = ModelInput(textTokenizer, mathTokenizer, trainData)