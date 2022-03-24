import numpy as np
from scipy.fftpack import fft
from tqdm import tqdm
import os
import librosa as lbs

sampleData = np.array([])
sampleMean = np.array([])
sampleNr = 0.
samples = os.listdir('C:/Users/Mathias Sebastian/PycharmProjects/SensoryReplacementUnit/Learning_Set/normalized')


def getfourier(filename):
    global sampleData
    global sampleNr
    global sampleMean
    data, fs = lbs.load(filename, sr=18000, mono=True)  # Read audio file and separate data
    a = data  # Take first track of the two sided signal
    b = [(ele / 2 ** 16.) * 2 - 1 for ele in a]  # Normalize 16-bit tracks on [-1, 1]
    c = fft(b)  # Calculate FFT
    d = int(len(c) / 2)  # Account for Nyquist Limit
    psd = abs(c[:(d-1)])**2 # Calculate Power Spectral Density

    sampleData = np.append(sampleData, psd) # Append PSD for later use
    sampleNr = sampleNr + 1.  # Increment total sample count
    if len(sampleMean) == 0:  # Check is PSD is the first one
        sampleMean = np.append(sampleMean, psd)
    else:                     # Add PSD to the mean calculation
        length = np.minimum(len(sampleMean), len(psd))
        for i in range(length):
            sampleMean[i] = sampleMean[i] + psd[i]

def getPSD(data):  #Calculate PSD
    mData = np.abs(data)**2
    return mData


print('Constructing Learning Set...')
for z in tqdm(range(0, len(samples))): # Get the PSD and add it to the mean calculation
    getfourier('Learning_Set/normalized/' + samples[z])

sampleMean = sampleMean / sampleNr  # Calculate final mean value
print(sampleMean)  # Print mean PSD for manual inspection
np.save('learning_set.npy', sampleMean)  # Save mean PSD in file
print('Done!')
