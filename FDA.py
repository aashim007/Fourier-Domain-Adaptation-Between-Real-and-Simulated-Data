#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install librosa


# In[ ]:


import tensorflow as tf
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import math
from scipy.linalg import svd
from scipy.signal import firwin, lfilter
import scipy.signal as signal
from scipy.fftpack import dct, idct

NumberOfSensors=32
SoundVelocity = 1500
ArrayUpperDesignFrequency = 4000
InterElementSpacing =SoundVelocity/(2*ArrayUpperDesignFrequency)


def init_waterfall_tf(n, nhistory, nsamples):
    # global wf
    wf = tf.experimental.numpy.zeros((n, nhistory+1, nsamples))
    return wf
def init_waterfall(n, nhistory, nsamples):
    # global wf
    wf = np.zeros((n-1, nhistory+1+1, nsamples))
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
    # global wf

    # wf[n, 2:nhistory + 1, :] = wf[n, 1:nhistory, :]
    wf[n-1, 2:nhistory + 1, :] = wf[n-1, 1:nhistory, :]

    wf[n-1, 1, :] = y

    if clim is None:
        plt.imshow(np.squeeze(wf[n-1, :, :]))
    else:
        plt.imshow(np.squeeze(wf[n-1, :, :]), clim=clim)

    plt.xlim([0, 180])
    plt.ylim([1, 40])

    plt.xticks([0, 30, 60, 90, 120, 150, 180])
    plt.yticks([1, 40])

wf=None
def WDPlot(BeamOutputPower, leg, m, n, p):
    SteeringVector = np.arange(0, 181, 1)
    ObservationTimeInSeconds = 40
    wf = init_waterfall(1, 300, len(SteeringVector))
    plt.subplot(m, n, p)
    n=1
    nhistory=300
    clim=None
    wf[n-1, 2-1:nhistory + 1+1, :] = wf[n-1, 1-1:nhistory+1, :]
    for j in range(ObservationTimeInSeconds):
        y = BeamOutputPower[:, j]
        wf[n-1, j, :] = y
        if clim is None:
            plt.imshow(np.squeeze(wf[n-1, :, :]))
        else:
            plt.imshow(np.squeeze(wf[n-1, :, :]), clim=clim)
        # disp_waterfall(wf, 1, 300, np.arange(1, len(SteeringVector) + 1), BeamOutputPower[:, j].T )#/ np.max(BeamOutputPower[:, j].T))
    plt.text(5, 15, leg, fontsize=10, color='magenta')# fontname='Helvetica',
    plt.xlim([0, 180])
    plt.ylim([1, 40])
    plt.xticks([0, 30, 60, 90, 120, 150, 180])
    plt.yticks([1, 40])
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Frequencies')

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
    SteeringVector = np.arange(0, 181)  # Changed from [0:1:180] to use NumPy's arange
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
    h[[id, id+1]] = hf  # FIR filter for interpolation
    
    xo = lfilter(h, 1, xi)
    
    return xo
def simulate_ULA(signal, TargetBearing, Bbf1, Bbf2, M, d, fs, c):
    # Calculate the target delay vector
    TauVector = np.arange(M) * (d * np.cos(np.deg2rad(TargetBearing)) / c)
    
    # Calculate sample delays and maximum sample delay
    SampleDelay = TauVector * fs
    MaximumSampleDelay = np.max(np.abs(SampleDelay))
    SampleDelay += MaximumSampleDelay
    
    s1 = np.zeros((len(signal), M))
    
    for i in range(M):
        s1[:, i] = delay(signal, SampleDelay[i], Bbf1 / 1000, Bbf2 / 1000, fs / 1000)
        s1[:, i] /= (np.std(s1[:, i]) + np.finfo(float).eps)
    
    return s1

def ADM(NumberOfSensors, TgtSNR, TargetBearing, BroadbandOnOff,
        BearingIncrementDegreesPerSeconds, SignalStartFrequency,
        SignalEndFrequency, SamplingFrequency, ObservationTimeInSeconds,
        A1, f1, A2, f2):
    BroadBandShapeFIRFilterOrder = 512
    
    # Initialize variables
    NumberOfSamples = SamplingFrequency + BroadBandShapeFIRFilterOrder
    SignalArrayDataMatrix1 = []
    m1, m2, fm1, fm2 = 0.5, 0.5, 5, 10
    #nyquist = 0.5*SamplingFrequency
    for frames in range(1, ObservationTimeInSeconds + 1):
        WhiteSignal = np.random.randn(NumberOfSamples)
        FIRFilterCoef = firwin(BroadBandShapeFIRFilterOrder, 2.0 * np.array([SignalStartFrequency, SignalEndFrequency]) / SamplingFrequency, pass_zero=False)
        Signal = lfilter(FIRFilterCoef, 1, WhiteSignal)
        TgtSignal = ((1 + (m1 * np.sin(2 * np.pi * fm1 / SamplingFrequency * np.arange(1, NumberOfSamples + 1))) +
                        (m2 * np.sin(2 * np.pi * fm2 / SamplingFrequency * np.arange(1, NumberOfSamples + 1)))) * Signal) * BroadbandOnOff + \
                    A1 * np.sin(2 * np.pi * f1 / SamplingFrequency * np.arange(1, NumberOfSamples + 1)) + \
                    A2 * np.sin(2 * np.pi * f2 / SamplingFrequency * np.arange(1, NumberOfSamples + 1))
        
        ArrayDataMatrix = simulate_ULA(TgtSignal, TargetBearing, SignalStartFrequency, SignalEndFrequency,
                                        NumberOfSensors, InterElementSpacing, SamplingFrequency, SoundVelocity)
        
        SignalArrayDataMatrix1.extend(ArrayDataMatrix[BroadBandShapeFIRFilterOrder:])
        TargetBearing += BearingIncrementDegreesPerSeconds
    
    SignalStd = 10 ** (TgtSNR / 20)
    SignalArrayDataMatrix = SignalStd * np.array(SignalArrayDataMatrix1)
    
    return SignalArrayDataMatrix

def gen_adm(sinr, tb, ObvTime, num_s=32):
    NumberOfSensors=num_s
    wf=None
    NoiseOnOff = 1
    SINR = sinr
    SteeringVector = np.arange(181)
    ObservationTimeInSeconds = ObvTime
    BearingIncrement = 0
    BB = 0

    # Initialize Ocean & Array parameters - Implement ArrayInitialization function here

    SamplingFrequency = 4096
    BroadBandShapeFIRFilterOrder = 512
    NumberOfSamples = SamplingFrequency + BroadBandShapeFIRFilterOrder
    SignalStartFrequency = 100
    SignalEndFrequency = 2000

    # %%%%%%%%%%%%%%%%%%%%%%%%%% Interference Signal %%%%%%%%%%%%%%%%%%%%%%%%%%
    InterferenceBearing = 15
    IntBroadbandOnOff = BB
    IntBearingIncrementDegreesPerSeconds = 0
    Ai1 = 1
    fi1 = 1200
    Ai2 = 0
    fi2 = 0
    InterferenceSNR = 20

    InterferenceArrayDataMatrix = ADM(NumberOfSensors, InterferenceSNR, InterferenceBearing, IntBroadbandOnOff,
                                        IntBearingIncrementDegreesPerSeconds, SignalStartFrequency, SignalEndFrequency,
                                        SamplingFrequency, ObservationTimeInSeconds, Ai1, fi1, Ai2, fi2)

    FIRFilterCoef = firwin(BroadBandShapeFIRFilterOrder, [2.0 * SignalStartFrequency / SamplingFrequency,
                                                            2.0 * SignalEndFrequency / SamplingFrequency], pass_zero=False)

    # Add noise to the interference signal
    le = len(InterferenceArrayDataMatrix)
    InterferenceNoiseArrayDataMatrix = np.zeros((le, NumberOfSensors))

    for i in range(NumberOfSensors):
        AmbientNoise = np.random.randn(le)
        AmbientNoise = lfilter(FIRFilterCoef, 1, AmbientNoise)
        AmbientNoise = AmbientNoise / np.std(AmbientNoise)
        InterferenceNoiseArrayDataMatrix[:, i] = InterferenceArrayDataMatrix[:, i] + NoiseOnOff * AmbientNoise

    # %%%%%%%%%%%%%%%%%%%%%%%%%% Target Signal %%%%%%%%%%%%%%%%%%%%%%%%%%
    TgtBroadbandOnOff = BB
    As1 = Ai1
#     fs1 = fi1
    fs1 = 1300
    As2 = Ai2
    fs2 = fi2
    TargetBearing = tb
    SignalSNR = SINR + InterferenceSNR

    SignalArrayDataMatrix = ADM(NumberOfSensors, SignalSNR, TargetBearing, TgtBroadbandOnOff,
                                BearingIncrement, SignalStartFrequency, SignalEndFrequency,
                                SamplingFrequency, ObservationTimeInSeconds, As1, fs1, As2, fs2)

    FinalArrayDataMatrix = SignalArrayDataMatrix + InterferenceNoiseArrayDataMatrix
    FinalArrayDataMatrix = FinalArrayDataMatrix / np.max(np.abs(FinalArrayDataMatrix))
    SignalArrayDataMatrix =SignalArrayDataMatrix/np.max(np.abs(SignalArrayDataMatrix))
    return FinalArrayDataMatrix, SignalArrayDataMatrix, InterferenceNoiseArrayDataMatrix


def main(keras_path, tflite_path, rt_data_path):
    def model(input):                               #defining autoencoder model    
        # Encoder
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        encoder = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        # Decoder
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(encoder)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2,1), activation="relu", padding="same")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        decoder = tf.keras.layers.Conv2D(1, (3, 3), activation="tanh", padding="same")(x)
        return decoder

    #FOURIER DOMAIN ADAPTIONs

    def low_freq_mutate_np(signal_src, signal_trg, L=0.1):
        # Compute the FFT of the source and target signals along the specified axis (axis 0)
        a_src = np.fft.fftshift(np.fft.fft(signal_src, axis=0), axes=0)
        a_trg = np.fft.fftshift(np.fft.fft(signal_trg, axis=0), axes=0)

        N, num_channels = signal_src.shape
        b = int(np.floor(N * L))

        # Calculate the range of low-frequency components
        h1 = N // 2 - b
        h2 = N // 2 + b + 1

        # Replace the low-frequency components of the source with the target along each channel
        for channel in range(num_channels):
            a_src[h1:h2, channel] = a_trg[h1:h2, channel]

        # Perform the inverse FFT to obtain the modified signal
        modified_signal = np.fft.ifft(np.fft.ifftshift(a_src, axes=0), axis=0)

        return modified_signal.real

    def FDA_source_to_target_signal(src_signal, trg_signal, L=0.1):
        # Compute the FFT of the source and target signals
        fft_src = np.fft.fftshift(np.fft.fft2(src_signal, axes=(0, 1)), axes=(0, 1))
        fft_trg = np.fft.fftshift(np.fft.fft2(trg_signal, axes=(0, 1)), axes=(0, 1))

        # Extract amplitude and phase of both FFTs
        amp_src, pha_src = np.abs(fft_src), np.angle(fft_src)
        amp_trg, pha_trg = np.abs(fft_trg), np.angle(fft_trg)

        # Mutate the amplitude part of source with target
        amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

        # Mutated FFT of source
        fft_src_ = amp_src_ * np.exp(1j * pha_src)

        # Get the mutated signal
        src_in_trg = np.fft.ifft2(np.fft.ifftshift(fft_src_, axes=(0, 1)), axes=(0, 1))
        src_in_trg = np.real(src_in_trg)

        return src_in_trg
    
    #FOR LOADING REAL-TIME DATA
    ls_rt_data=[]
    audio_ls=[]
    for parent_folder in os.listdir(rt_data_path):
        joined_path = os.path.join(rt_data_path, parent_folder)
        for zoom_folder in os.listdir(joined_path):
            joined_path1 = os.path.join(joined_path, zoom_folder)
            for file in os.listdir(joined_path1):
                final_path = os.path.join(joined_path1, file)
                if file.endswith('.wav') or file.endswith('.WAV') or file.endswith('.mp3'):
                    audio, _ = librosa.load(final_path, sr=4096)
                    audio_ls.append(audio)
            audio1 = audio_ls[0]
            audio2 = audio_ls[1]
            audio3 = audio_ls[2]
            audio4 = audio_ls[3]
            audio_ls=[]
            num_segments = len(audio1)//4096
            for i in range(num_segments):
                start = i * 4096
                end = start + 4096
                segment1 = audio1[start:end]
                segment2 = audio2[start:end]
                segment3 = audio3[start:end]
                segment4 = audio4[start:end]
                f_seg = np.stack((segment1, segment2, segment3, segment4), axis=1)
                ls_rt_data.append(f_seg)
    ls_rt_data = np.asarray(ls_rt_data)
    print(ls_rt_data.shape)
    
    #LOAD SIMULATED DATA
    ls_SINR = []
    for i in range(-30, 1, 1):
        ls_SINR.append(i)
    ls_targetBearing = []
    for i in range(30, 151, 2):
        ls_targetBearing.append(i)
    lsFADM=[]
    lsSDM=[]
    ls_L=[0,0.01,0.05,0.1,0.25,0.5,1]
    arr = np.arange(0, 1891, step=1)
    ObvTime=1
    for sinr in ls_SINR:
        for tb in ls_targetBearing:
            FinalArrayDataMatrix, SignalArrayDataMatrix, InterferenceNoiseArrayDataMatrix = gen_adm(sinr, tb, ObvTime, num_s=4)
            lsFADM.append(FinalArrayDataMatrix)
            lsSDM.append(SignalArrayDataMatrix)
    FADM = np.asarray(lsFADM)
    SDM = np.asarray(lsSDM)
    
    #USING FDA FUNCTIONS DEFINED ABOVE (APPLYING FDA)
    F_ls_train=[]
    F_ls_label=[]
    for j in ls_L:
        for i in range(len(ls_rt_data)):
            k = np.random.choice(arr, replace=False)
            F_ls_train.append(FDA_source_to_target_signal(FADM[k], ls_rt_data[i], L=j))#0.1
            F_ls_label.append(SDM[k])
    
    #FINAL FDA DATA FOR TRAINING
    FADM = np.asarray(F_ls_train)
    SDM = np.asarray(F_ls_label)
    
    #TRAIN-TEST SPLIT DATA
    X_train, X_val, y_train, y_val = train_test_split(FADM, SDM, shuffle=True, test_size=0.1, random_state=123)
    
    #BUILDING MODEL AND DEFINING INPUT LAYER
    model_inputs = tf.keras.layers.Input(shape=(X_train.shape[1],X_train.shape[2],1))
    model_output = model(model_inputs)
    model = tf.keras.Model(inputs=model_inputs, outputs=model_output)
    
    model.compile(optimizer="Adam", loss=tf.keras.losses.MeanSquaredError()) #run_eagerly=True
    model.summary()
    
    def step_decay(epoch):
        initial_lrate = 0.0001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,  
               math.floor((1+epoch)/epochs_drop))
        return lrate
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,verbose=1, mode='min')]
    my_lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)
    
    #TRAINING
    t0 = time.time()
    result=model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_val, y_val), callbacks=[callbacks, my_lr_scheduler])
    print("Training time:", time.time()-t0)

    #PLOT LOSS-CURVE
    figsize=(10,9)
    figure,ax=plt.subplots(figsize=figsize)
    plt.plot(result.history['loss'],label='Training Loss',linewidth=3)
    plt.plot(result.history['val_loss'],label='Validation Loss',linewidth=3)
    plt.legend(loc='upper right',fontsize=17)
    plt.xlabel('Epoch',fontsize=22)
    plt.ylabel('loss',fontsize=22)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    #plt.savefig("C:/Users/hlakh/Desktop/NRB_Demonstration/outputs/Supervised_loss.png")
    plt.show()

    #SAVING MODEL
    path = keras_path
    model.save(path)
    model = tf.keras.models.load_model(path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tf.io.write_file(tflite_path, tflite_model)
    
#Enter model paths and real-tme data path here
keras_path="F:\\ml\\keras.h5"
tflite_path = "F:\\ml\\tf.tflite"
rt_data_path="F:\\ml\\real_time_data"  
main(keras_path, tflite_path, rt_data_path)


# In[ ]:





# In[ ]:




