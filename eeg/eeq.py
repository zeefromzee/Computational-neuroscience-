import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import unittest
import pytest

class eeg:
    def __init__(self):

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data = pd.read_csv(os.path.join(self.script_dir, "eeg.csv"), sep=";", decimal=",")
        self.subset = self.data[['HH', 'MM', 'SS', 'EOG Left', 'EEG C3-A1', 'EEG O1-A1', 'EEG C4-A1','EEG O2-A1']]

    def subset_desc(self):    
        print(self.subset.describe())

    def convert_fft(self):
        sampling_rate = 200
        column_array = self.subset['EEG C3-A1'].to_numpy()
        fre1 = scipy.fft.fft(column_array)
        self.magnitude = np.abs(fre1)
        self.frequency = scipy.fft.fftfreq(len(column_array), d=1/sampling_rate)
        print(scipy.fft.fft(column_array))

    def create_graph(self):
        if not hasattr(self, 'self.frequency'):
            raise Exception("covert to fft first`   ")
        plt.plot(self.frequency, self.magnitude)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()

    def convert_stft(self, channel):        
        co_array = self.subset[channel].to_numpy()
        self.f, self.t, self.Zxx = scipy.signal.stft(co_array, fs=200, nperseg=400)       
        
    def encode_vec(self):
        if not hasattr(self, 'Zxx'):
            raise Exception("convert_stft must be called before encode_vec")
        ampl_matrix = np.abs(self.Zxx)
        flat = ampl_matrix.flatten()
        # divide by its L2 norm
        norm = np.linalg.norm(flat)
        self.psi = flat/norm
        if flat.all() < 1e-10 :
            print("The norm is 0")
        else:
            self.psi = flat/norm
        
    def compute_fidelity(self, other):
        inner_prod = np.dot(self.psi, other.psi) # second object's psi vector(other.psi)
        F = np.abs(inner_prod) ** 2
        return np.clip(F, 0, 1)
    
class TestEEG(unittest.TestCase):

    def setUp(self):
        # this runs before every single test automatically
        # create your session objects here once
        self.session1 = eeg()
        self.session2 = eeg()

    def test_psi_unit_norm(self):
        # arrange and act
        self.session1.convert_stft('EEG C3-A1')
        self.session1.encode_vec()
        # assert
        norm = np.linalg.norm(self.session1.psi)
        self.assertAlmostEqual(norm, 1.0, places=10)

    def test_fidelity_symmetric(self):
        return 0

if __name__ == '__main__':
    unittest.main()

session1 = eeg()
session2 = eeg()

session1.convert_stft('EEG C3-A1')
session1.encode_vec()
session1.convert_fft()
session1.create_graph()

session2.convert_stft('EEG C4-A1')
session2.encode_vec()

fidelity = session1.compute_fidelity(session2)
print(fidelity)
