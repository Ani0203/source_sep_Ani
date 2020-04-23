import soundfile as sf
import argparse
import musdb
import museval
import test
import multiprocessing
import functools
from pathlib import Path
import torch
import json
import tqdm
import numpy as np
import os
import pickle
import crepe
import scipy.io.wavfile
import mir_eval
import essentia.standard as ess
from matplotlib import pyplot as plt
#%matplotlib inline

################################################################
#DEFINE FUNCTIONS

#Obtain SDR values for each test example as well as save estimate audios
# Define required functions
def pad_or_truncate(
    audio_reference,
    audio_estimates
):
    """Pad or truncate estimates by duration of references:
    - If reference > estimates: add zeros at the and of the estimated signal
    - If estimates > references: truncate estimates to duration of references

    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    Returns
    -------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    """
    est_shape = audio_estimates.shape
    ref_shape = audio_reference.shape
    if est_shape[1] != ref_shape[1]:
        if est_shape[1] >= ref_shape[1]:
            audio_estimates = audio_estimates[:, :ref_shape[1], :]
        else:
            # pad end with zeros
            audio_estimates = np.pad(
                audio_estimates,
                [
                    (0, 0),
                    (0, ref_shape[1] - est_shape[1]),
                    (0, 0)
                ],
                mode='constant'
            )

    return audio_reference, audio_estimates

def evaluate(
    references,
    estimates,
    win=1*44100,
    hop=1*44100,
    mode='v4',
    padding=True
):
    """BSS_EVAL images evaluation using metrics module

    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    window : int, defaults to 44100
        window size in samples
    hop : int
        hop size in samples, defaults to 44100 (no overlap)
    mode : str
        BSSEval version, default to `v4`
    Returns
    -------
    SDR : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    ISR : np.ndarray, shape=(nsrc,)
        vector of Source to Spatial Distortion Image (ISR)
    SIR : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    SAR : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    """

    estimates = np.array(estimates)
    references = np.array(references)

    if padding:
        references, estimates = pad_or_truncate(references, estimates)

    SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(
        references,
        estimates,
        compute_permutation=False,
        window=win,
        hop=hop,
        framewise_filters=(mode == "v3"),
        bsseval_sources_version=False
    )

    return SDR, ISR, SIR, SAR

#######################################################################################    
#Create folder within given experiment corresponding to model being evaluated
exp_no = 8

#model = '../new_models/test/model_tabla_mse_pretrain1/' #Path to the saved model
#model_name = 'model_tabla_mse_pretrain1'

model_names = ['model_tabla_mse_pretrain4' , 'model_tabla_bce_westerntrainable_finetune_3_1' , 'model_tabla_bce_westerntrainable_finetune_4_1' , 'model_tabla_bce_ourmixtrainable_finetune_3_1' , 'model_tabla_bce_ourmixtrainable_finetune_4_1' ] 

for model_name in model_names: 
    model = '../new_models/' + model_name + '/'
    print("TESTING: ", model_name)


    exp_model_res_path = '../test_out/Exp_' + str(exp_no) + '/' + model_name + '/'
    
    if not(os.path.exists(exp_model_res_path)):
        os.mkdir(exp_model_res_path)
        
        
    #Create folders corresponding to each test examples 
    test_data_folder = '../rec_data_final/test/'
    for test_eg in os.listdir(test_data_folder):
        test_eg_path = exp_model_res_path + test_eg + '/'
        if not(os.path.exists(test_eg_path)):
            os.mkdir(test_eg_path)
            
    
    
    
    #targets = ['vocals']
    targets = ['tabla']
    root = '../rec_data_final/'
    subset = 'test'
    cores = 1
    no_cuda = False
    is_wav = True
    samplerate = 44100
    use_cuda = not no_cuda and torch.cuda.is_available()
    print("USE_CUDA: ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    mus = musdb.DB(
        root=root,
        download=root is None,
        subsets=subset,
        is_wav=is_wav
    )
    
    
    # iterate over all tracks present in test folder
    for track in mus.tracks:
        outdir = exp_model_res_path + track.name + '/'    
        print(track.name)
        estimates = test.separate(
            audio=track.audio,
            targets=targets,
            model_name=model,
            niter=2,
            alpha=1,
            softmask=False,
            device=device
        )    
        
        for target, estimate in estimates.items():
            sf.write(
                outdir / Path(target).with_suffix('.wav'),
                estimate,
                samplerate
            )
        
        print("SAVED SEPARATED VOCALS AND ACCOMPANIMENTS!")
        
        audio_estimates = []
        audio_reference = []
        eval_targets = []
        
        for key, target in list(track.targets.items()):
           try:
               # try to fetch the audio from the user_results of a given key
               estimates[key]
           except KeyError:
               # ignore wrong key and continue
               continue
           eval_targets.append(key)
          
        mode='v4'
        win=1.0
        hop=1.0
        data = museval.aggregate.TrackStore(win=win, hop=hop, track_name=track.name)
        
        # check if vocals and accompaniment is among the targets
        #has_acc = all(x in eval_targets for x in ['vocals', 'accompaniment'])
        has_acc = all(x in eval_targets for x in ['tabla', 'accompaniment'])
        if has_acc:
           # remove accompaniment from list of targets, because
           # the voc/acc scenario will be evaluated separately
           eval_targets.remove('accompaniment')
          
        #audio_estimates.append(estimates['vocals'])
        #audio_reference.append(track.targets['vocals'].audio)
        
        audio_estimates.append(estimates['tabla'])
        audio_reference.append(track.targets['tabla'].audio)
        
        
        SDR, ISR, SIR, SAR = evaluate(
               audio_reference,
               audio_estimates,
               win=int(win*track.rate),
               hop=int(hop*track.rate),
               mode=mode
           )
        
        save_dict = {}
        save_dict['SDR'] = SDR[0].tolist()
        save_dict['ISR'] = ISR[0].tolist()
        save_dict['SDR_median'] = np.median(SDR[0])
        save_dict['ISR_median'] = np.median(ISR[0])
        save_dict['SIR'] = SIR[0].tolist()
        save_dict['SAR'] = SAR[0].tolist()
        save_dict['SIR_median'] = np.median(SIR[0])
        save_dict['SAR_median'] = np.median(SAR[0])
        
        
        
        save_file = outdir + "evaluation.json"
        
        print("Saving json file")
        
        with open(save_file, 'w') as outfile:
           json.dump(save_dict, outfile)
        
        print("Saved!")