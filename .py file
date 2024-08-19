import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import math
from scipy import signal
import tensorflow as tf

# Constants
NumberOfSensors = 32
SoundVelocity = 1500
ArrayUpperDesignFrequency = 4000
InterElementSpacing = SoundVelocity / (2 * ArrayUpperDesignFrequency)

# Initialization functions
def init_waterfall_tf(n, nhistory, nsamples):
    wf = tf.experimental.numpy.zeros((n, nhistory + 1, nsamples))
    return wf

def init_waterfall(n, nhistory, nsamples):
    wf = np.zeros((n - 1, nhistory + 2, nsamples))
    return wf

def int_fil(f1, f2, fs, del_val):
    w = np.pi * (f2 - f1) / fs
    w1 = np.pi * (f2 + f1) / fs
    w3 = w * del_val
    w4 = w1 * del_val
    
    r = np.array([[1, np.cos(w1) * np.sin(w) / w],
                  [np.cos(w1) * np.sin(w) / w, 1]])
    
    coef = np.linalg.inv(r).dot(np.array([np.cos(w4) * np.sin(w3 + np.finfo(float).eps) / (w3 + np.finfo(float).eps),
                                           np.cos(w1 - w4) * np.sin(w - w3 + np.finfo(float).eps) / (w - w3 + np.finfo(float).eps)]))
    return coef

def int_fil_tf(f1, f2, fs, del_val):
    w = tf.constant(tf.math.pi * (f2 - f1) / fs, dtype=tf.float32)
    w1 = tf.constant(tf.math.pi * (f2 + f1) / fs, dtype=tf.float32)
    w3 = w * del_val
    w4 = w1 * del_val

    r = tf.constant([[1.0, tf.math.cos(w1) * tf.math.sin(w) / w],
                     [tf.math.cos(w1) * tf.math.sin(w) / w, 1.0]], dtype=tf.float32)
    
    coef = tf.linalg.solve(r, [tf.math.cos(w4) * tf.math.sin(w3) / (w3 + tf.math.finfo(tf.float32).eps),
                               tf.math.cos(w1 - w4) * tf.math.sin(w - w3) / (w - w3 + tf.math.finfo(tf.float32).eps)])
    
    return coef

def disp_waterfall(wf, n, nhistory, x, y, clim=None):
    wf[n - 1, 2:nhistory + 1, :] = wf[n - 1, 1:nhistory, :]
    wf[n - 1, 1, :] = y

    if clim is None:
        plt.imshow(np.squeeze(wf[n - 1, :, :]))
    else:
        plt.imshow(np.squeeze(wf[n - 1, :, :]), clim=clim)

    plt.xlim([0, 180])
    plt.ylim([1, 40])
    plt.xticks([0, 30, 60, 90, 120, 150, 180])
    plt.yticks([1, 40])

def WDPlot(BeamOutputPower, leg, m, n, p):
    SteeringVector = np.arange(0, 181, 1)
    ObservationTimeInSeconds = 40
    wf = init_waterfall(1, 300, len(SteeringVector))
    plt.subplot(m, n, p)
    n = 1
    nhistory = 300
    clim = None
    wf[n - 1, 2 - 1:nhistory + 1 + 1, :] = wf[n - 1, 1 - 1:nhistory + 1, :]
    for j in range(ObservationTimeInSeconds):
        y = BeamOutputPower[:, j]
        wf[n - 1, j, :] = y
        if clim is None:
            plt.imshow(np.squeeze(wf[n - 1, :, :]))
        else:
            plt.imshow(np.squeeze(wf[n - 1, :, :]), clim=clim)
        plt.text(5, 15, leg, fontsize=10, color='magenta')
    plt.xlim([0, 180])
    plt.ylim([1, 40])
    plt.xticks([0, 30, 60, 90, 120, 150, 180])
    plt.yticks([1, 40])
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Frequencies')

# Signal processing functions
def DS_4096(ArrayDataMatrix):
    NumberOfSensors = 4
    SamplingFrequency = 12800
    SoundVelocity = 1500
    ArrayUpperDesignFrequency = 4000
    InterElementSpacing = SoundVelocity / (2 * ArrayUpperDesignFrequency)
    ProcessingStartFrequency = 100
    ProcessingEndFrequency = 2000
    LMMSEStartFrequency = ProcessingStartFrequency
    LMMSEEndFrequency = ProcessingEndFrequency

    b, a = signal.butter(6, [2 * ProcessingStartFrequency / SamplingFrequency, 2 * ProcessingEndFrequency / SamplingFrequency], 'bandpass')
    ArrayDataMatrix = signal.lfilter(b, a, ArrayDataMatrix, axis=0)
    ArrayDataMatrix = ArrayDataMatrix.T  # Transpose for row vector

    NumberOfSamples = len(ArrayDataMatrix[0])
    SteeringVector = np.arange(0, 181)
    SampleOffset = int(np.ceil((NumberOfSensors * InterElementSpacing * SamplingFrequency) / SoundVelocity))

    BeamOutput = np.zeros((len(SteeringVector), NumberOfSamples))

    for k in range(len(SteeringVector)):
        Tau = InterElementSpacing * -1 * np.cos(np.deg2rad(SteeringVector[k])) / SoundVelocity
        PartialBeamOutput = np.zeros(NumberOfSamples)

        for i in range(NumberOfSensors):
            ActualDelay = (Tau * (i - 1) * SamplingFrequency) + SampleOffset
            IntegerSampleDelay = int(np.fix(ActualDelay))
            FractionalSampleDelay = ActualDelay - IntegerSampleDelay

            LMMSEFilterCoef = int_fil(LMMSEStartFrequency / 1000, LMMSEEndFrequency / 1000, SamplingFrequency / 1000, FractionalSampleDelay)
            FIRFilterCoef = np.zeros(IntegerSampleDelay + 3)
            FIRFilterCoef[[IntegerSampleDelay + 1, IntegerSampleDelay + 2]] = LMMSEFilterCoef
            FilteredOutput = signal.lfilter(FIRFilterCoef, 1, ArrayDataMatrix[i, :])
            PartialBeamOutput += FilteredOutput

        BeamOutput[k, :] = PartialBeamOutput

    init_waterfall(1, 300, len(SteeringVector))
    BeamOutputPower = np.zeros((len(SteeringVector), NumberOfSamples // SamplingFrequency))
    BeamOutput1 = BeamOutput.T
    BeamOutputEnergy = BeamOutput1 ** 2
    TotalEnergy = np.sum(BeamOutputEnergy, axis=0)

    for k in range(len(SteeringVector)):
        for j in range(1, int(np.floor(NumberOfSamples / SamplingFrequency)) + 1):
            start = (j - 1) * SamplingFrequency
            stop = j * SamplingFrequency
            BeamOutputPower[k, j - 1] = (1 / SamplingFrequency) * np.sum(BeamOutputEnergy[start:stop, k])

    return TotalEnergy, BeamOutputPower, BeamOutput1

def delay(xi, d, f1, f2, fs):
    id = int(np.floor(d))  # Integer delay
    fd = d - id  # Fractional delay
    hf = int_fil(f1, f2, fs, fd)  # Interpolation filter for fractional delay
    
    h = np.zeros(id + 2)
    h[[id, id + 1]] = hf  # FIR filter for interpolation
    
    xo = signal.lfilter(h, 1, xi)
    
    return xo

def simulate_ULA(signal, TargetBearing, Bbf1, Bbf2, M, d, fs, c):
    TauVector = np.arange(M) * (d * np.cos(np.deg2rad(TargetBearing)) / c)
    SampleDelay = TauVector * fs
    MaximumSampleDelay = np.max(np.abs(SampleDelay))
    SampleDelay += MaximumSampleDelay
    
    s1 = np.zeros((len(signal), M))
    
    for i in range(M):
        s1[:, i] = delay(signal, SampleDelay[i], Bbf1 / 1000, Bbf2 / 1000, fs / 1000)
        s1[:, i] /= (np.std(s1[:, i]) + np.finfo(float).eps)
    
    return s1

def ADM(NumberOfSensors, TgtSNR, TargetBearing, BroadbandOnOff, BearingIncrementDegreesPerSeconds, SignalStartFrequency, SignalEndFrequency, SamplingFrequency, ObservationTimeInSeconds, A1, f1, A2, f2):
    BroadBandShapeFIRFilterOrder = 512
    NumberOfSamples = SamplingFrequency + BroadBandShapeFIRFilterOrder
    SignalArrayDataMatrix1 = []
