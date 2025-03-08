import argparse
import random
import torch
from fmcib.models import fmcib_model

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class FCIBUptune(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fmcib_model().to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_labels = 2
        self.classifier_dropout = torch.nn.Dropout(config['dropout_prob'])
        self.classifier_interim = torch.nn.Linear(config['fcib_output_size'], config['head_interim_size'])
        self.interim_af = F.gelu
        self.classifier_head = torch.nn.Linear(config['head_interim_size'], config['num_labels'])

  def forward(self, input):
      output = self.model(input)
      output_dropout = self.classifier_dropout(output)
      classifier_interim = self.classifier_interim(output_dropout)
      interim_activated = self.interim_af(classifier_interim)
      classifier_output = self.classifier_head(interim_activated)
      return classifier_output
