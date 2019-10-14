#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:40:24 2019

@author: aniruddha
"""

#Extracts pitch track for clean vocals and extracted vocals using, saves both as csv files in corresponding folder
#Saves plot of time vs. frequency of both together

import crepe
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt

track_folder = '../test_out/Exp_1/musdb_eg1/'

clean_vocs_path = track_folder + 'vocals_clean.wav'
extr_vocs_path = track_folder + 'vocals.wav'

r1, clean_vocs = scipy.io.wavfile.read(clean_vocs_path)
r2, extr_vocs = scipy.io.wavfile.read(extr_vocs_path)

t_clean_vocs, f_clean_vocs, conf_clean_vocs, act_clean_vocs = crepe.predict(clean_vocs, r1, viterbi=True)
t_extr_vocs, f_extr_vocs, conf_extr_vocs, act_extr_vocs = crepe.predict(extr_vocs, r2, viterbi=True)

if (f_clean_vocs.shape[0]>f_extr_vocs.shape[0]):
    f_clean_vocs = f_clean_vocs[:f_extr_vocs.shape[0]]
    t_clean_vocs = t_extr_vocs
elif (f_clean_vocs.shape[0]<=f_extr_vocs.shape[0]):
    f_extr_vocs = f_extr_vocs[:f_clean_vocs.shape[0]]
    t_extr_vocs = t_clean_vocs

clean_pitch_track =  np.transpose(np.concatenate((np.atleast_2d(t_clean_vocs), np.atleast_2d(f_clean_vocs)), axis=0))
extr_pitch_track =  np.transpose(np.concatenate((np.atleast_2d(t_extr_vocs), np.atleast_2d(f_extr_vocs)), axis=0))

np.savetxt(track_folder+'clean_pitch.csv', clean_pitch_track, delimiter=',')
np.savetxt(track_folder+'extr_pitch.csv', extr_pitch_track , delimiter=',')


# =============================================================================
# for i in range(conf_clean_vocs.shape[0]):
#     if conf_clean_vocs[i]<0.5:
#         f_clean_vocs[i]=0
#     else:
#         pass
# 
# for i in range(conf_extr_vocs.shape[0]):
#     if conf_extr_vocs[i]<0.5:
#         f_extr_vocs[i]=0
#     else:
#         pass
# =============================================================================

mse = np.mean(np.square(f_clean_vocs - f_extr_vocs))


plt.plot(t_clean_vocs, f_clean_vocs)
plt.plot(t_clean_vocs, f_extr_vocs)

plt.legend(['Clean vocals', 'Extracted vocals'], loc='upper left')
#plt.title('MSE='+str(mse), loc='left')

#plt.show()
plt.savefig(track_folder+'pitch_track_crepe.pdf')