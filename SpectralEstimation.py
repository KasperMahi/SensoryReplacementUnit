import numpy as np
import pyaudio
from scipy.fftpack import fft
from scipy import signal
import math
import serial

print('Initializing audio stream...')
print(' ')
FORMAT = pyaudio.paInt16  # Audio Format
CHANNELS = 1  # Number of Audio Channels
RATE = 18000  # Microphone sample rate (Hz)
CHUNK = 900  # Number of samples per millisecond
SAMPLE_TIME = 0.1  # Sample time in seconds
SAMPLE_TIME_MILLIS = 1  # Sample Time in Millis
pa = pyaudio.PyAudio()  # Initialize PyAudio
isRunning = True  # Initialize isRunning Boolean variable
learningSet = np.load("learning_set.npy", allow_pickle=True)  # Load learning set from file
ser = serial.Serial('COM7', baudrate=9600, timeout=1)





def main():
    global isRunning
    stream = pa.open(format=FORMAT,  # Open audio stream
                     channels=CHANNELS,
                     input=True,
                     rate=RATE,
                     frames_per_buffer=CHUNK)
    stream.start_stream()  # Start streaming
    print('Done!')
    print('Detecting ...')
    while isRunning:  # Continuous loop to keep estimation process running
        try:
            Block = stream.read(CHUNK)  # Read recorded audio segment of CHUNK Samples
            #Block = lowpass(Block)
            estimate(Block, CHUNK)  # Run Spectral Estimation algorithm on the recorded Block of CHUNK samples
        except KeyboardInterrupt:  # Termination condition
            isRunning = False


def getpowerspectrum(data):
    # █ ------------------------------------------------------------ █
    # █ Function to calculate the Power Spectral Density of a signal █
    # █ Data is the sample data of the signal to be processed        █
    # █ ------------------------------------------------------------ █

    c = fft(data)  # Calculate the Fourier Transform of the Discrete Time Signal
    ps = (abs(c)) ** 2  # Calculate Power Spectral Density
    return ps


def estimate(data, sr):
    # █ ---------------------------------------------------------------- █
    # █ Main function to estimate the similarity of the recorded spectra █
    # █ to the mean of the learning set.                                 █
    # █ Data is the sample data of the recorded signal and sr is the     █
    # █ number of samples.                                               █
    # █ ---------------------------------------------------------------- █

    global isRunning  # Tells function to use global variable
    global CHUNK

    specData = getpowerspectrum(np.frombuffer(data, np.int16))  # Calculate Power Spectral Density of recorded signal

    for k in range(1, int(len(learningSet) / CHUNK)):  # Calculate IS Distance for each section of the Learning set
        isData = getisd(learningSet, specData, sr * (k - 1), sr * k, sr)
        if isData > 33000:  # Detection Threshold
            ser.write(b'k')
            print('█ Detected █')
            print(isData)


def getisd(spec1, spec2, srDown, srUp, sr):
    # █ -------------------------------------------------------------------------- █
    # █ Function to calculate the Itakura-Saito Distance between two Power Spectra █
    # █ Spec1 is the mean power spectrum of a Learning set.                        █
    # █ Spec2 is the power spectrum of the input sound.                            █
    # █ srDown, srUp and sr are the lower limit, upper limit and total samples.    █
    # █ -------------------------------------------------------------------------- █

    p1 = 1 / sr * (sum(spec1[srDown:srUp:1])) ** 2  # Normalize Spectra 1
    sp1 = spec1[srDown:srUp:1] / math.sqrt(p1)

    p2 = 1 / sr * (sum(spec2[:srUp])) ** 2  # Normalize Spectra 2
    sp2 = spec2[:srUp] / math.sqrt(p2)

    c = 1e-3  # Control value to ensure no division by zero

    mData = sum(
        sp2 / (sp1 + c) - np.log(sp2 / (sp1 + c) + c) - 1)  # Calculate the Itakura Saito distance in one direction
    nData = sum(
        sp1 / (sp2 + c) - np.log(sp1 / (sp2 + c) + c) - 1)  # Calculate the Itakura Saito distance in other direction
    rData = (mData + nData) / 2  # Take the average to make the calculation symmetric
    return rData


def lowpass(spec):
    b, a = signal.butter(5, 100, "low", analog=True, output='ba')
    output = signal.filtfilt(b, a, spec)
    return output


main()
