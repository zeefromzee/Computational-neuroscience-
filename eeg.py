import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
# import array
# Date = pd.date_range("08032016", periods=6)
# df = pd.DataFrame(np.random.randn(6, 4), index=Date, columns=list("EOG Left"))
script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, "eeg.csv"), sep=";", decimal=",")
subset = data[['HH', 'MM', 'SS', 'EOG Left', 'EEG C3-A1', 'EEG O1-A1', 'EEG C4-A1','EEG O2-A1']]
print(subset.describe())

s = pd.Series(np.random.randn(8), index=['HH', 'MM', 'SS', 'EOG Left', 'EEG C3-A1', 'EEG O1-A1', 'EEG C4-A1','EEG O2-A1'])
s1 = pd.Series(np.random.randn(4), index=['HH', 'MM', 'SS', 'EOG Left'])
# print(s1.array)
print(subset['EEG C3-A1'].shape)
column_array = subset['EEG C3-A1'].to_numpy()
print(" ")
print(column_array)

sampling_rate = 200
fre1 = scipy.fft.fft(column_array)
magnitude = np.abs(fre1)
frequency = scipy.fft.fftfreq(248440, d=1/sampling_rate)
print(scipy.fft.fft(column_array))

plt.plot(frequency, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()