#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code downloads the data for reproducing the results of the original article
"""

import context
import numpy as np
import urllib.request
import zipfile
import os

size = 512                 # Network size
R = 1000000                # Repetitions of the simulation
H0 = 0.5                   # Uniform distribution of fields parameter
J0 = 1.0                   # Average value of couplings
Js = 0.1                   # Standard deviation of couplings


# Download data from the asymmetric SK model
B=21
betas = 1 + np.linspace(-1, 1, B) * 0.3
for ib in range(B):
    beta_ref = round(betas[ib], 3)
    filename = 'data-H0-' + str(H0) + '-J0-' + str(J0) + '-Js-' + str(
        Js) + '-N-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    url = 'https://zenodo.org/record/4318983/files/'+filename
    print('Beginning download with urllib2 of '+ url)

    urllib.request.urlretrieve(url, 'data/'+filename)
    
# Download data for the forward, inverse, and phase reconstruction problems
files = ['forward.zip', 'inverse.zip', 'reconstruction.zip']
for f in files:
    url = 'https://zenodo.org/record/4318983/files/'+f
    print('Beginning download with urllib2 of '+ url)
    urllib.request.urlretrieve(url, 'data/'+f)
    with zipfile.ZipFile('data/'+f, 'r') as zip_ref:
        zip_ref.extractall('data/')
    os.remove('data/'+f)

