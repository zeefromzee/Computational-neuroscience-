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
