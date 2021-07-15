# -*- coding: utf-8 -*-
"""

ICBHI 2017 Challenge - Respiratory Sound Database
    - https://bhichallenge.med.auth.gr/ - 
    
Script to produce a classification of healthy vs non-healthy patients based on 
analysis of audio files recorded from various patients using different 
acquisition methods.

Main steps are:
    1. identify .wav files on HDD, produce useful dataset overview based on 
    information from .wav header and filename structure
    2. read audio from .wav files and check signal in time and spectral domains
    3. downcovert audio to match lowest sampling rate
    4. prepare classifier to predict healthy v non-healthy subjects


@author: michael@eikonal.uk


versions of libraries:
---------------------
Python          3.8.5
matplotlib      3.3.1
numpy           1.19.2
scipy           1.5.2
sklearn         0.23.2
imblearn        0.8.0
---------------------
"""


import re
import pandas as pd
import wave
import numpy as np
#import imblearn

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from os import listdir
from collections import Counter

from scipy.fft import fft, rfft, fftfreq, rfftfreq
from scipy.signal import stft, resample, resample_poly
from scipy.linalg import svd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve, plot_precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import RandomOverSampler as ros



plt_fig = 0  #change to 1 to plot figures of audio, spectra etc

# get data....downloaded files to HDD for speed of access
file_path = r'C:\Users\mikey\Documents\Python\audio\ICBHI\ICBHI_final_database'
filename_list = listdir(file_path)

#create seperate wav and text filenames for use later
filename_wav = [f for f in filename_list if f.endswith(".wav")]
filename_txt = [f for f in filename_list if f.endswith(".txt")]

# get diagnosis/classification data
fn_diagnosis = 'ICBHI_Challenge_diagnosis.txt'
fn_trainTest = 'ICBHI_challenge_train_test.txt'

num_files = len(filename_wav)

diag_path = file_path + '\\' + fn_diagnosis
trainTest_path = file_path + '\\' + fn_trainTest

df = pd.read_csv(diag_path, sep=" ", delimiter = "\t", header=None)
df.columns = ['patient no.','diagnosis']

#for index, row in df.iterrows():
#    if row['diagnosis'] != 'Healthy':
#        df['diagnosis'][index] = 'Unhealthy'
    

for i in range(len(df)):
    if df['diagnosis'][i] != 'Healthy':
        df['diagnosis'][i] = 'Unhealthy'


#%% get train/test data from file and write to dataframe
# sort out healthy/unhealthy labels in df


df_trainTest = pd.read_csv(trainTest_path, sep=" ", delimiter = "\t", header=None)
df_trainTest.columns = ['filename','train-test']


# match the patient name with health/unhealthy
S = df_trainTest['filename']
df_diagnosis = df_trainTest 
d = ['empty']*num_files
df_diagnosis['diagnosis'] = d


count = 0
for i in range(len(df)):
    found = []
    fill=[]
    tofind = str(df['patient no.'][i])
    found = S[S.str.match(tofind) == True]
    fill = df.iloc[i]['diagnosis']

    for j in range(len(found)):
        df_diagnosis.loc[count+j, 'diagnosis'] = fill

    jlen = len(found)
    count = count + jlen



#%% analyse number of patients, number of measurement locations etc, etc
# there are 5 parts to the filename

filename_parts= [0]*num_files
filename_patientnumber= [0]*num_files
filename_recordingindex = [0]*num_files
filename_chestlocation = [0]*num_files
filename_daqmode = [0]*num_files
filename_recequip = [0]*num_files

# get info from filenames
for i in range(num_files):
    filename_parts[i] = re.split('[_.]+', filename_wav[i])
    filename_patientnumber[i] = filename_parts[i][0]
    filename_recordingindex[i] = filename_parts[i][1]
    filename_chestlocation[i] = filename_parts[i][2]
    filename_daqmode[i] = filename_parts[i][3]
    filename_recequip[i] = filename_parts[i][4]

#count the various parameters
count_patientnumber = Counter(filename_patientnumber)
count_recordingindex = Counter(filename_recordingindex)
count_chestlocation = Counter(filename_chestlocation)
count_daqmode = Counter(filename_daqmode)
count_recequip = Counter(filename_recequip)

# quick plot of data features to establish balance of dataset
# create 2x2 subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Analysis of dataset from filename parts')

# actual plots
# patient numbers
axs[0,0].scatter(count_patientnumber.keys(), count_patientnumber.values(), marker='x', s=10)
axs[0,0].tick_params('x', labelrotation=90)
axs[0,0].set_xlabel("# of measurements per patient")
axs[0,0].xaxis.set_major_locator(tkr.MultipleLocator(5))

# measurement location
axs[0,1].bar(count_chestlocation.keys(), count_chestlocation.values())
axs[0,1].set_xlabel("measurement location")

# data acquisition mode
axs[1,0].bar(count_daqmode.keys(), count_daqmode.values())
axs[1,0].set_xlabel("acquisition mode")

# measuring equipement
axs[1,1].bar(count_recequip.keys(), count_recequip.values())
axs[1,1].set_xlabel("acquisition equipment")
axs[1,1].tick_params('x', labelrotation=45)


fig.tight_layout()
plt.show()


print('CONCLUSIONS OF DATA SCOPING:', '\n',
      '----------------------------', '\n',
      ' - Number of measurements per patient varies considerably, some patients have 1 measurement whilst others have up to 60','\n',
      ' - Measurement locations are reasonably balanced, only "lateral left" appears to be lower than the others', '\n',
      ' - Multichannel acquisition mode is used more than single = ',count_daqmode['mc'], 'vs', count_daqmode['sc'],'\n',
      ' - Acquisition equipment most used is the AKGC417L', '\n', '\n',
      )


#%% analyse wav files
filename_params = [] 
nchannel = []
sampwid = []
framerate = []
nframes = []
comtype = []
comname = []


for i in range(num_files):
    w = wave.open(file_path + '//' + filename_wav[i], mode='rb')
    filename_params.append(w.getparams())
    nchannel.append(filename_params[i].nchannels)
    sampwid.append(filename_params[i].sampwidth)
    framerate.append(filename_params[i].framerate)
    nframes.append(filename_params[i].nframes)
    comtype.append(filename_params[i].comptype)
    comname.append(filename_params[i].compname)
    #print(w.getparams())
    
    
    
print('DISCUSSION AROUND .WAV FILE HEADERS:', '\n',
      '----------------------------', '\n', 
      'number of channels used are:...', set(nchannel), '\n',
      'sample widths used are:........', set(sampwid) ,'\n',
      'frame rates used are:..........', set(framerate),'\n',
      #'number of frames used are:....', set(nframes), '\n',
      'comtype used is:...............', set(comtype),'\n',
      'compression used is:...........', set(comtype),'\n','\n',
      'Assumptions made from this are:', '\n',
      ' - that the .wav files collected in "mc" mode are interleaved, de-interleaving is outwidth the scope of this project', '\n',
      ' - sample widths of 2 and 3 bytes are assumed to be 16 and 24 bit audio respectively', '\n',
      ' - there are multiple numbers of frames used (not shown) - wont effect final analysis', '\n',
      ' - the variable frame rates will affect the frequency content of the data, a method to deal with this will be required', '\n','\n'
      )



#%%  functions for reading in wav files and check signals

# read wav files and return the audio signal ampltides versus time
def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file.
    (c) warren weckesser (wavio.py - githubGist) (readwav & _wav2array) - used 
    this to handle the sample width differences
    """
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        #raw_bytes = np.fromstring(data, dtype=np.uint8)
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
        
    return result


def readwav(file):
    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    
    return rate, sampwidth, array

#%% quick visualisation of amplitude time and spectra

lowest_framerate = min(framerate)

# visualise the audio filies in amplitude v time
def visualise_audio(path: str, fn: str):
    
    frame_rate, samp_width, audio_signal = readwav(path +'\\' +  fn)
    
    print(" frame rate is:....", frame_rate, "Hz", "\n",
          "frame time is:....", "{:3.3f}".format((1/frame_rate *1e6)), "us"
          )

    # time = frame rate / size of signal
    audio_time = np.linspace((1/frame_rate),len(audio_signal) / frame_rate, num=len(audio_signal))
    
    print(" audio time is:.......", audio_time.max(), "sec", '\n',
          "number of samples is:......", len(audio_signal)/1e3, "ks",'\n','\n'
          )
    
    if plt_fig == 1:    #set above in 1st few lines
        
        plt.figure()
        plt.plot(audio_time, audio_signal)
        plt.title(fn + 'amplitude v time')
        plt.show()
        
        plt.figure()
        yf = rfft(audio_signal) #plot real fft of audio signal
        #print('yf is', len(yf))
        num_samples = len(audio_signal)
        #print(num_samples)
        #xf = rfftfreq(num_samples, d=1/frame_rate)
        xf = 0.5/audio_time #rfftfreq not working just use inverse of time * 2 nyquist
        #print('xf is', len(xf))
        
        plt.plot(xf, np.abs(yf))
        plt.title(fn + 'spectra')
        plt.show()
    
    
    down_rate = int(frame_rate / lowest_framerate)  #select minimum frame rate for downsampling
    
    # due to time restrictions used this down sampling approach - would conduct
    # more accurate downsampling for future application
    #down_samples = round(len(audio_signal) * float(down_rate) / frame_rate)
    audio_new = resample_poly(audio_signal, 1, 10)
    time_new = resample_poly(audio_time, 1, 10)
    
    return audio_new, time_new
    
# uncomment this if you want an amplitude array of all samples
#a_out = []
#for i in range(num_files):
    #path = 'C://Users//mikey//Documents//Python//audio//ICBHI//ICBHI_final_database' + '//'+ filename_wav[i]
#    print(filename_wav[i])
#    anew, tnew = visualise_audio(file_path, filename_wav[i])
#    a_out.append(anew)
    #plt.plot(tnew, anew)
    #plt.show()
    

print('CONCLUSIONS FROM AUDIO DATA:', '\n',
      '************************************', '\n', 
      'Randonly selected some files to look at audio amplitudes', '\n',
      'the variable length of recording would make loading the raw audio into','\n',
      'a classifier (e.g. SVM) difficult unless equalising the sampling count was conducted.','\n',
      'This is out of the scope of this quick test but could be achieved in the future.','\n','\n',
      'As the frame rates vary a method of downsampling should be applied','\n',
      'scipy resample_poly was used which is not best method for audio, but suitable for this test.','\n',
      'For future application would use more effective downsampling to be used ', '\n','\n',
      )

#%% conduct short time ft on data, downsample to min frame rate for comparison
# calculate the energy mean and the svd to simplify/reduce dimensionality for input into classifier

lowest_framerate = min(framerate)

def visualise_audio_stft(path: str, fn: str):
    
    #audio_signal = []
    #audio_new = []
    #frame_rate = []
    #samp_width = []
    
    #reading the audio files as np  - float
    frame_rate, samp_width, asig = readwav(path +'\\' +  fn)
    asigout=[]
    for x in asig: asigout.append(x)
    asx = [float(i) for i in asigout]
    audio_signal = np.array(asx)
    
    # = np.concatenate(asigout)
    #audio_input = wave.open(path +'\\' +  fn, mode='rb')
    #read all the frames
    #audio_signal = audio_input.readframes(-1)
    #audio_signal = np.frombuffer(audio_signal, dtype="int32")
    
    #frame_rate = audio_input.getframerate()
    print(" frame rate is:....", frame_rate, "Hz", '\n',
          "frame time is:....", "{:3.3f}".format((1/frame_rate *1e6)), "us"
          )
    
    # time = frame rate / size of signal
    audio_time = np.linspace((1/frame_rate),len(audio_signal) / frame_rate, num=len(audio_signal))
    
    print(" audio time is:.......", audio_time.max(), "sec", '\n',
          "number of samples is:......", len(audio_signal)/1e6, "Ms"
          )

    # --downsample all data to lowest framerate
    down_rate = int(frame_rate / lowest_framerate)  #select minimum frame rate for downsampling
    
    #down_samples = round(len(audio_signal) * float(down_rate) / frame_rate)
    
    audio_new = resample_poly(audio_signal, 1, down_rate)
    #time_new = resample_poly(audio_time, 1, 10)
       
    num_samples = len(audio_new)
    #yf = rfft(audio_signal)
    #frame_rate = lowest_framerate  #update frame rate for fft
    
    f, t, Zxx = stft(audio_new, lowest_framerate)
    #f, t, Zxx = stft(audio_signal, frame_rate)
    
    #plt.figure()
    #plt.pcolormesh(t,f, np.abs(Zxx), vmin=0, vmax=5,shading='gouraud')
    #plt.pcolormesh(t,f, np.abs(Zxx), shading='gouraud')
    #plt.title(fn + 'spectra')
    #plt.show()
    
    return f, t, Zxx

# reduce dimensionality of audio_stft matrix to vector

#  calculate the mean of Zxx
z_mean=[]
#  calculate the SVD of Zxx
z_svd=[]


for i in range(num_files):   #num_files
    #path = file_path + '//'+ filename_wav[i]
    print(filename_wav[i])
    # get spectra
    f, t, Zxx = visualise_audio_stft(file_path, filename_wav[i])
    #calculate spectral mean
    z_abs = np.abs(Zxx)
    #z_abs = np.mean(np.abs(Zxx),axis=0)
    z_out = np.mean(z_abs,axis=1)
    z_mean.append(z_out)
    
    #calculate 1d from svd 
    U, s, vt = svd(Zxx)
    z_svd.append(s)
    
    plt.plot(z_mean[i])
    plt.title(df_diagnosis['diagnosis'][i])
    plt.show()



#%% Train a classifier to predict the diagnosis of the healthy -v unhealthy signals
# if time permits compare different classifiers and different classification modes (e.g. time, spectra....)

print('--------FURTHER WORK: preprocessing--------------- ', '\n'
      ' 1. try Mel-freqs Cepstral Coefficients ', '\n'   
      ' 2. wavelet transforms ', '\n'
      ' 3. go through audio and split further based on coughs/wheezes, also include background noise', '\n'
      ' 4. ', '\n'
      ' ', '\n'
      ' ', '\n'
      )

# find indexes of test and train
ind_test = []
ind_train = []

for i in range(len(df_trainTest)):
    if df_trainTest['train-test'][i] == 'test':
        ind_test.append(df_trainTest.index[i])
    elif df_trainTest['train-test'][i] == 'train':
        ind_train.append(df_trainTest.index[i])

ltest = len(ind_test)
ltrain = len(ind_train)

# calculate percent test to train
ltt = ltest + ltrain
pctest = round(ltest/ltt*100)
pctrain = round(ltrain/ltt*100)

print("number of test samples is:..........", ltest, '\n'
      "number of training  samples is:.....", ltrain, '\n'
      "ratio of training to test is........", pctrain, ':', pctest,'\n','\n'
      )

# convert classes to binary 
class_change = {'Healthy': 1, 'Unhealthy': 0}
y = [class_change[item] for item in df_diagnosis['diagnosis']]

# count binary types (i.e # of each class)
num_h = sum(y)
num_u = num_files - num_h

print("ratio of healthy v non-healthy is.........", num_h, ':', num_u, '\n','\n'
      "It is clear from this ratio that the classification dataset is skewed in favour of unhealthy.", '\n'
      "This will impact the effectiveness of training an accurate classifier.  Possible solutions are", '\n'
      "Oversampling healthy or undersampling unhealthy.  In this instance try random oversampling"
      )

# create X & y for classification 

#******UNCOMMENT TO CHANGE BETWEEN z_mean and z_svd****************
#--------------------------------------------------------------------
#X = np.stack(z_svd, axis=0)  #create array from list, patient feature rows
X = np.stack(z_mean, axis=0)  #create array from list, patient feature rows

# rebalance dataset with oversampling
oversample = ros(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)

print("check oversampler worked,", Counter(y_over))


scaler = MinMaxScaler()
scaler.fit(X_over)
X_norm = scaler.transform(X_over)

#X_train = np.zeros(len(ind_train))
#y_train = np.zeros(len(ind_train))
#X_test = np.zeros(len(ind_test))
#y_test = np.zeros(len(ind_test))
#y = np.array(y_over, dtype=np.float32)

y = y_over

# X_train = X_norm[ind_train]
# X_test  = X_norm[ind_test]
# y_train = y[ind_train]
# y_test = y[ind_test]

# now that dataset has been equalised can use train/test/split: keep original 60/40
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.4, random_state=42)



#%% setup linear SVclassification using cross val to split train/test data x times

# set up linear hyperplane with a relatively large soft-margin (C=1)
margin = 10  #try 0.1, 1, 10 etc
clf_its = 5  #try out different numbers of cross_val iterations (e.g. 2 to 10)
clf = SVC(kernel='linear', C=margin, random_state=42)
#clf = LinearSVC(C=margin)
scores_svc = cross_val_score(clf, X_train, y_train, scoring="accuracy", cv=clf_its)
y_train_svc_prediction = cross_val_predict(clf, X_train, y_train, cv=clf_its)

print("--------------SVC CLASSIFIER------------------------", '\n'
      "running cross validation %d times on training data gives fitting scores of:" % clf_its, "\n"
      " ",  scores_svc, "\n"
      "...with %0.3f mean accuracy and standard deviation of %0.4f" % (scores_svc.mean(), scores_svc.std()), "\n"
      "this is using a soft-margin for fitting of %0.2f: " %margin
      )

#### now fit predicted to training dataset
clf.fit(X_train, y_train)

svc_prediction = clf.predict(X_test)
svc_accuracy = accuracy_score(svc_prediction, y_test)

print("accuracy of the SVM test classification is: %0.3f" %svc_accuracy, "\n", "\n"
      "Therefore, comparing the SVM training and test accuracy we have: \n"
      "Training accuracy = %0.3f" %scores_svc.mean(), "\n"
      "Test accuracy     = %0.3f" %svc_accuracy)

#confuse_matrix = confusion_matrix(y_test, y_prediction)
confuse_matrix_svc = confusion_matrix(y_train, y_train_svc_prediction)

precision_svc = precision_score(y_train, y_train_svc_prediction)
recall_svc = recall_score(y_train, y_train_svc_prediction)
f1_svc = f1_score(y_train, y_train_svc_prediction)

print("-----------------------------------------------------------\n",
      "Confusion matrix for SVC model is: \n", confuse_matrix_svc, "\n"
      " Precision is equal to: ", "%0.6f" %precision_svc, "\n"
      " Recall is equal to: ", "%0.6f" %recall_svc, "\n"
      " F1 score is equal to: ", "%0.6f" %f1_svc
      )

disp = plot_precision_recall_curve(clf, X_test, y_test)



#%% setup DecisionTreeClassifier using cross val to split train/test data x times

maxDepth = 5 # can change from None to 1-sample number (n-1), although not recommended
clf = DecisionTreeClassifier(max_depth=maxDepth, random_state=42)
clf_its = 5
scores_dtc = cross_val_score(clf, X_train, y_train, scoring="accuracy", cv=clf_its)
y_train_prediction_dtc = cross_val_predict(clf, X_train, y_train, cv=clf_its)


print("--------------DECISION TREE CLASSIFIER------------------------", '\n'
      "running cross validation %d times gives fitting scores of:" % clf_its, "\n"
      " ",  scores_dtc, "\n"
      "...with %0.2f mean accuracy and standard deviation of %0.2f" % (scores_dtc.mean(), scores_dtc.std()), "\n"
      " using a maximum depth setting of: %0.0f" %maxDepth
      )

#
clf.fit(X_train, y_train)

prediction_dtc = clf.predict(X_test)
accuracy_dtc = accuracy_score(prediction_dtc, y_test)

print("Accuracy of the Decision Tree test classification is: %0.3f" %accuracy_dtc, "\n\n",
      "Therefore, comparing Decision Tree training and test accuracy we have: \n"
      "Training accuracy = %0.3f" %scores_dtc.mean(), "\n"
      "Test accuracy     = %0.3f" %accuracy_dtc)

#confuse_matrix = confusion_matrix(y_test, y_prediction)
confuse_matrix_dtc = confusion_matrix(y_train, y_train_prediction_dtc)

precision_dtc = precision_score(y_train, y_train_prediction_dtc)
recall_dtc = recall_score(y_train, y_train_prediction_dtc)
f1_dtc = f1_score(y_train, y_train_prediction_dtc)

print("-----------------------------------------------------------------", "\n"
      "Confusion matrix for DecisionTree model is: \n", confuse_matrix_dtc, "\n"
      " Precision is equal to: ", "%0.3f" %precision_dtc, "\n"
      " Recall is equal to: ", "%0.3f" %recall_dtc, "\n"
      " F1 score is equal to: ", "%0.3f" %f1_dtc
      )

disp = plot_precision_recall_curve(clf, X_test, y_test)


#### Conclusion:

print(
""" 
CLASSIFIER CONCLUSIONS:  ,
Manually tested the z_mean vs the z_svd  which were vectors of the Zxx (output from stft).
It appears that z_mean is slightly better for classification purposes.

By simple comparison of the confusion matrix of the two 
classifiers it appears that the Decision Tree classifier is better at predicting 
healthy vs unhealthy patients due to its smaller size of false positives and 
false negatives.  This is also shown in the higher levels of prediction 
accuracy from the training/test sets."""
)


# def main():
#     #plots on or off
#     plt_fig = 0  #change to 1 to plot figures of audio, spectra etc
    
#     #print end of program
#     print('........classification complete............')




# if __name__ == '__main__':

#------------------------------------------------------------
    __author__ = "Michael Robinson"
    __copyright__ = "Copyright 2021, M Robinson"
    __license__ = "GPL"
    __version__ = "0.0.1"
    __maintainer__ = "M Robinson"
    __status__ = "Prototype"






