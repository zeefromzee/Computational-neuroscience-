import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import unittest
import pytest
from abc import ABC,abstractmethod

class eeg(ABC):
    def __init__(self):

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data = pd.read_csv(os.path.join(self.script_dir, "eeg.csv"), sep=";", decimal=",")
        self.subset = self.data[['HH', 'MM', 'SS', 'EOG Left', 'EEG C3-A1', 'EEG O1-A1', 'EEG C4-A1','EEG O2-A1']]

    def subset_desc(self):    
        print(self.subset.describe())

    @abstractmethod
    def convert_fft(self, channel):
        """Abstract method - must be implemented by subclass"""
        pass
    
    def create_graph_fft(self):
        # if not hasattr(self, 'self.frequency'):
        #     raise Exception("covert to fft first`   ")
        plt.plot(self.frequency, self.magnitude)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()

    def create_graph_stft(self):
        amp = 2 * np.sqrt(2)
        plt.pcolormesh(self.t, self.f, np.abs(self.Zxx), vmin=0, vmax=amp, shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    @abstractmethod
    def convert_stft(self, channel):
        """Abstract method - must be implemented by subclass"""
        pass

    def encode_vec(self):
        if not hasattr(self, 'Zxx'):
            raise Exception("convert_stft must be called before encode_vec")
        # Flatten STFT (keep complex values - don't use np.abs)
        flat = self.Zxx.flatten()
        # Compute L2 norm (works for complex numbers too)
        norm = np.linalg.norm(flat)
        
        if norm < 1e-10:
            print("WARNING: norm is near zero")
            self.psi = flat  # Store as-is
        else:
            self.psi = flat / norm

    def compute_fidelity(self, other):
        """Compute quantum fidelity: F = |<psi1|psi2>|^2"""
        inner_prod = np.dot(self.psi, np.conj(other.psi))
        F = np.abs(inner_prod) ** 2
        return np.clip(F, 0, 1)


class EEGSession(eeg):

    def convert_fft(self,channel):
        sampling_rate = 200
        column_array = self.subset[channel].to_numpy()
        fre1 = scipy.fft.fft(column_array)
        self.magnitude = np.abs(fre1)
        self.frequency = scipy.fft.fftfreq(len(column_array), d=1/sampling_rate)
        print(scipy.fft.fft(column_array))
    
    def convert_stft(self, channel):        
        co_array = self.subset[channel].to_numpy()
        self.f, self.t, self.Zxx = scipy.signal.stft(co_array, fs=200, nperseg=400)


class SimilarityMetrics:

    @staticmethod
    def euclidean_distance(psi1, psi2):
        
        return np.linalg.norm(psi1 - psi2)
    
    @staticmethod
    def cosine_similarity(psi1, psi2):
        
        return np.dot(psi1, psi2) / (np.linalg.norm(psi1) * np.linalg.norm(psi2))
    
    @staticmethod
    def fidelity(psi1, psi2):
        """Quantum fidelity: F = |<psi1|psi2>|^2 (works for complex vectors)"""
        inner_prod = np.dot(psi1, np.conj(psi2))
        return np.abs(inner_prod) ** 2


class AnomalyDetector:
    
    def __init__(self, metric_name='fidelity'):
        self.metric_name = metric_name
        self.baseline = None
        
    def set_baseline(self, psi_vectors_list):
        self.baseline = np.mean(psi_vectors_list, axis=0)
        self.baseline = self.baseline / np.linalg.norm(self.baseline)
    
    def compute_score(self, psi):
        if self.baseline is None:
            raise Exception("Baseline not set. Call set_baseline() first")
        
        if self.metric_name == 'fidelity':
            return SimilarityMetrics.fidelity(psi, self.baseline)
        elif self.metric_name == 'cosine':
            return SimilarityMetrics.cosine_similarity(psi, self.baseline)
        elif self.metric_name == 'euclidean':
            # For euclidean, convert to similarity (1 - normalized distance)
            dist = SimilarityMetrics.euclidean_distance(psi, self.baseline)
            return 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unknown metric: {self.metric_name}")
    
    def detect(self, psi, threshold):
        score = self.compute_score(psi)
        return score < threshold, score


class EvaluationMetrics:
    
    
    @staticmethod
    def precision_recall_f1(y_true, y_pred):

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    
    @staticmethod
    def roc_auc(y_true, y_scores):
        
        # Sort by scores
        sorted_indices = np.argsort(-y_scores)
        y_true_sorted = y_true[sorted_indices]
        
        # Count positives and negatives
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        # Calculate TPR and FPR for each threshold
        tp = np.cumsum(y_true_sorted) / n_pos
        fp = np.cumsum(1 - y_true_sorted) / n_neg
        
        # AUC is area under TPR vs FPR curve
        auc = np.trapz(tp, fp)
        return auc
    

class TestEEG(unittest.TestCase):

    def setUp(self):
        self.session1 = EEGSession()
        self.session2 = EEGSession()

    def test_psi_unit_norm(self):
        # arrangin
        self.session1.convert_stft('EEG C3-A1')
        self.session1.encode_vec()
        norm = np.linalg.norm(self.session1.psi)
        self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_similarity_metrics(self):
        
        self.session1.convert_stft('EEG C3-A1')
        self.session1.encode_vec()
        
        self.session2.convert_stft('EEG C4-A1')
        self.session2.encode_vec()
        
        # All metrics should be in valid ranges
        fidelity = SimilarityMetrics.fidelity(self.session1.psi, self.session2.psi)
        cosine = SimilarityMetrics.cosine_similarity(self.session1.psi, self.session2.psi)
        euclidean = SimilarityMetrics.euclidean_distance(self.session1.psi, self.session2.psi)
        
        self.assertTrue(0 <= fidelity <= 1)
        self.assertTrue(-1 <= cosine <= 1)
        self.assertTrue(euclidean >= 0)


# if __name__ == '__main__':
#     unittest.main()



baseline1 = EEGSession()
baseline2 = EEGSession()

baseline1.convert_stft('EEG C3-A1')
baseline1.encode_vec()

baseline2.convert_stft('EEG C4-A1')
baseline2.encode_vec()

# test sesion
session1 = EEGSession()
session2 = EEGSession()

session1.convert_stft('EEG C3-A1')
session1.encode_vec()
session1.convert_fft('EEG C3-A1')

session2.convert_stft('EEG O2-A1')
session2.encode_vec()
session2.convert_fft('EEG O2-A1')




fid = SimilarityMetrics.fidelity(session1.psi, session2.psi)
cos = SimilarityMetrics.cosine_similarity(session1.psi, session2.psi)
euc = SimilarityMetrics.euclidean_distance(session1.psi, session2.psi)

print(f"Fidelity:           {fid:.4f}")
print(f"Cosine Similarity:  {cos:.4f}")
print(f"Euclidean Distance: {euc:.4f}")


print("detect anomalies")


detectors = {
    'fidelity': AnomalyDetector('fidelity'),
    'cosine': AnomalyDetector('cosine'),
    'euclidean': AnomalyDetector('euclidean')
}

baseline_vectors = [baseline1.psi, baseline2.psi]

for metric_name, detector in detectors.items():
    detector.set_baseline(baseline_vectors)
    is_anomaly, score = detector.detect(session1.psi, threshold=0.5)
    print(f"{metric_name:12s} | Score: {score:.4f} | Anomaly (thresh=0.5): {is_anomaly}")


# session2.create_graph_fft()
# session2.create_graph_stft()