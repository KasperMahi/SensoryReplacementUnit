import numpy as np
import time
import pyaudio
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import librosa as lbs

FORMAT = pyaudio.paInt16  # Audio Format
CHANNELS = 1  # Number of Audio Channels
RATE = 18000  # Microphone sample rate (Hz)
CHUNK = 1800  # Number of samples per millisecond
SAMPLE_TIME = 0.1  # Sample time in seconds
SAMPLE_TIME_MILLIS = 1  # Sample Time in Millis
pa = pyaudio.PyAudio()
isRunning = True
now = time.time()
learningSet = np.load("learning_set.npy", allow_pickle=True)
soundData, fs = lbs.load('Learning_Set/normalized/1.wav', sr=18000, mono=True)  # Read audio file and separate data
#print(learningSet)


def main():
    global isRunning
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     input=True,
                     rate=RATE,
                     frames_per_buffer=CHUNK)

    while isRunning:
        try:
            stream.start_stream()
            Block = stream.read(CHUNK)
            #stream.stop_stream()
            runCorr(Block, CHUNK)
            #isRunning = False
        except KeyboardInterrupt:
            isRunning = False
        except:
            pass


def getfourier(data):
    now = time.time()
    b = [(ele / 2) * 2 - 1 for ele in data]  # Normalize 16-bit tracks on [-1, 1]
    c = fft(data)
    d = int(len(c) / 2)  # Account for Nyquist Limit
    ps = abs(c[:(d - 1)])**2  # Calculate PSD
    #print('FFT Time')
    #print(time.time() - now)
    return ps


def corrCoef(file1, file2, n):  # Returns the Normalized Correlation Coefficient of the two files
    sum = 0.
    sum1 = 0.
    sum2 = 0.
    for num in range(0, n - 1, 1):
        sum = sum + (file1[num] * file2[num])

    for enum in range(0, n - 1, 1):
        sum1 = sum1 + file1[enum] * file1[enum]
        sum2 = sum2 + file2[enum] * file2[enum]
    sum = sum / np.sqrt(sum1 * sum2)
    return sum


def runCorr(data, sr):
    global isRunning
    fftData = getfourier(np.frombuffer(data, np.int16))

    now = time.time()
    isData = getISDist(fftData, fftData, sr)
    print(isData)
    if isData < 100000:
        print('█████████████')
        print(isData)
    #isRunning = False


def getPSD(data):
    now = time.time()
    mData = np.abs(data)**2
    #print('PSD time')
    #print(time.time() - now)
    return mData


def getISDist(spec1, spec2, sr):
    # now = time.time()
    sp1Data = spec1[:sr]
    sp2Data = spec2[:sr]
    c = 1e-50 # Control value to ensure no division by zero
    mData = sum(sp2Data+c / sp1Data+c - np.log(sp2Data+c / sp1Data+c) - 1)
    #nData = sum(sp1Data+c / sp2Data+c - np.log(sp1Data+c / sp2Data+c) - 1)
    #rData = 0.5*(mData + nData)
    return mData


main()
