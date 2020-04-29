import argparse
import model
import model_mtl
import data
import torch
import time
from pathlib import Path
import tqdm
import json
import utils
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import copy
import librosa
from matplotlib import pyplot as plt


tqdm.monitor_interval = 0


def train(args, unmix, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    mse_losses = utils.AverageMeter()
    bce_losses = utils.AverageMeter()
    bce_sf_losses = utils.AverageMeter()
    #ADDED
    precision_values = utils.AverageMeter()
    recall_values = utils.AverageMeter()
    f_score_values = utils.AverageMeter()
    unmix.train()
    
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet) 
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        Y_hat, onset_probs  = unmix(x)
        Y = unmix.transform(y)
        #Y.shape==[255,16,2,2049]


        #Average spectrograms over stereo channels for SF onset detection
        Y_avg = Y.mean(dim=2)
        Y_hat_avg = Y_hat.mean(dim=2)

        Y_avg, Y_hat_avg = Y_avg.to(device), Y_hat_avg.to(device)

        # Compute SF proability vectors 
        #Feed log mel spectrograms to onset detection 
        loss_od = torch.zeros([Y.shape[1]]) #loss of size=batch size
        loss_od_sf = torch.zeros([Y.shape[1]])
        #ADDED
        prec = torch.zeros([Y.shape[1]])
        rec = torch.zeros([Y.shape[1]])
        f_sc = torch.zeros([Y.shape[1]])

        criterion1 = torch.nn.BCELoss()
        criterion2 = torch.nn.MSELoss()
                
        for x in range(Y.shape[1]):
            #loss_od[x] = criterion1(onset_probs[x], spectral_flux(magStft=Y_avg[:,x], bands=[0, (unmix.sample_rate)/2.0 ], fs=unmix.sample_rate, device=device))
            #loss_od_sf[x] = criterion1(spectral_flux(magStft=Y_hat_avg[:,x], bands=[0, (unmix.sample_rate)/2.0], fs=unmix.sample_rate, device=device), spectral_flux(magStft=Y_avg[:,x], bands=[0, (unmix.sample_rate)/2.0 ], fs=unmix.sample_rate, device=device))
            odf_targ = spectral_flux_vectorised_torch(magStft=Y_avg[:,x], device=device)
            odf_pred = spectral_flux_vectorised_torch(magStft=Y_hat_avg[:,x], device=device)
            odf_targ_np = odf_targ.detach().cpu().numpy()
            odf_pred_np = odf_pred.detach().cpu().numpy()
            #loss_od[x] = criterion1(onset_probs[x], spectral_flux_vectorised_torch(magStft=Y_avg[:,x], device=device))
            #loss_od_sf[x] = criterion1(spectral_flux_vectorised_torch(magStft=Y_hat_avg[:,x], device=device), spectral_flux_vectorised_torch(magStft=Y_avg[:,x], device=device))
            loss_od[x] = criterion1(onset_probs[x], odf_targ)
            loss_od_sf[x] = criterion1(odf_pred, odf_targ)
            prec[x], rec[x], f_sc[x] = eval_novelty(odf_targ=odf_targ_np, odf_pred=odf_pred_np)
            
# =============================================================================
# =============================================================================
#         plt.figure()
#         plt.title("Target novelty curve")
#         a = detect_onset(Y_mel_spect_chunks[0])
#         plt.plot(a.cpu().numpy())
#         plt.show()
#         print("SET OF 15 frames")
#         
#         plt.figure()
#         plt.title("Predicted Novelty curve")
#         a = detect_onset(Y_hat_mel_spect_chunks[0])
#         plt.plot(a.detach().cpu().numpy())
#         plt.show()
#         print("SET OF 15 frames")
# =============================================================================
        
        
# =============================================================================
#         for i in range (Y_mel_spect_chunks.shape[1]):   
#             print(i)
#             plt.figure(figsize=(5,20))
#             plt.imshow((Y_mel_spect_chunks.cpu().numpy()[0][i][0]), origin='lower')
#             plt.show()
#             
# =============================================================================
        
# =============================================================================
#         b = []
#         for i in range(Y_mel_spect_chunks.shape[1]):
#             c = detect_onset((Y_mel_spect_chunks[0][i])[None, :, :,:])
#             #print("HIII", c)
#             b.append(c[0][0])
#         
#         plt.figure()
#         plt.plot(np.array(b))
#         plt.show()
# =============================================================================
        
        
        loss_od = loss_od.to(device)
        loss_od_sf = loss_od_sf.to(device)
        #print("loss_size", loss_od.shape)
        
        
        
        mse_loss = criterion2(Y_hat, Y)
        bce_loss = (torch.sum(loss_od)/Y.shape[1])
        bce_sf_loss = (torch.sum(loss_od_sf)/Y.shape[1])
        #ADDED
        precision = (torch.sum(prec)/Y.shape[1])
        recall = (torch.sum(rec)/Y.shape[1])
        f_score = (torch.sum(f_sc)/Y.shape[1])
        
        #print ("MSE LOSS = ", mse_loss)
        #print("BCE_LOSS = ", bce_loss)
        #print("BCE SF LOSS = ", bce_sf_loss)
        
        #loss = (args.gamma)*bce_loss + (1-args.gamma)*mse_loss
        #loss = (1-args.gamma)*mse_loss
        #loss = ((args.gamma)/(2.0))*bce_loss + (1-args.gamma)*mse_loss + ((args.gamma)/(2.0))*bce_sf_loss
        loss = (1-args.gamma)*mse_loss + (args.gamma)*bce_sf_loss
        
        #print("TOTAL LOSS = ", loss)
        
        #float(Y_mel_spect_chunks.shape[0])
        
        
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
        mse_losses.update(mse_loss.item(), Y.size(1))
        bce_losses.update(bce_loss.item(), Y.size(1))
        bce_sf_losses.update(bce_sf_loss.item(), Y.size(1))
        #ADDED
        precision_values.update(float(precision), Y.size(1))
        recall_values.update(float(recall), Y.size(1))
        f_score_values.update(float(f_score), Y.size(1))
        #print("YOOO", type(losses.avg), type(recall_values.avg))
    return losses.avg , mse_losses.avg , bce_losses.avg, bce_sf_losses.avg, precision_values.avg, recall_values.avg, f_score_values.avg


def valid(args, unmix, device, valid_sampler, j, target_path):
    losses = utils.AverageMeter()
    mse_losses = utils.AverageMeter()
    bce_losses = utils.AverageMeter()
    bce_sf_losses = utils.AverageMeter()
    #ADDED
    precision_values = utils.AverageMeter()
    recall_values = utils.AverageMeter()
    f_score_values = utils.AverageMeter()
    unmix.eval()
    # Checking
    i = 0
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            Y_hat, onset_probs = unmix(x)
            Y = unmix.transform(y)
            
            #Average spectrograms over stereo channels for SF onset detection
            Y_avg = Y.mean(dim=2)
            Y_hat_avg = Y_hat.mean(dim=2)

            Y_avg, Y_hat_avg = Y_avg.to(device), Y_hat_avg.to(device)

            # Compute SF proability vectors 
            #Feed log mel spectrograms to onset detection 
            loss_od = torch.zeros([Y.shape[1]]) #loss of size=batch size
            loss_od_sf = torch.zeros([Y.shape[1]])
            
            #ADDED
            prec = torch.zeros([Y.shape[1]])
            rec = torch.zeros([Y.shape[1]])
            f_sc = torch.zeros([Y.shape[1]])
            
            criterion1 = torch.nn.BCELoss()
            criterion2 = torch.nn.MSELoss()
            
            #print("HEYYY", Y.shape, i)

            for x in range(Y.shape[1]):
                #loss_od[x] = criterion1(onset_probs[x], spectral_flux(magStft=Y_avg[:,x], bands=[0, (unmix.sample_rate)/2.0 ], fs=unmix.sample_rate, device=device))
                #loss_od_sf[x] = criterion1(spectral_flux(magStft=Y_hat_avg[:,x], bands=[0, (unmix.sample_rate)/2.0], fs=unmix.sample_rate, device=device), spectral_flux(magStft=Y_avg[:,x], bands=[0, (unmix.sample_rate)/2.0 ], fs=unmix.sample_rate, device=device))

                #loss_od[x] = criterion1(onset_probs[x], spectral_flux_vectorised_torch(magStft=Y_avg[:,x], device=device))
                #loss_od_sf[x] = criterion1(spectral_flux_vectorised_torch(magStft=Y_hat_avg[:,x], device=device), spectral_flux_vectorised_torch(magStft=Y_avg[:,x], device=device))
                odf_targ = spectral_flux_vectorised_torch(magStft=Y_avg[:,x], device=device)
                odf_pred = spectral_flux_vectorised_torch(magStft=Y_hat_avg[:,x], device=device)
                odf_targ_np = odf_targ.detach().cpu().numpy()
                odf_pred_np = odf_pred.detach().cpu().numpy()
                loss_od[x] = criterion1(onset_probs[x], odf_targ)
                loss_od_sf[x] = criterion1(odf_pred, odf_targ)
                prec[x], rec[x], f_sc[x] = eval_novelty(odf_targ=odf_targ_np, odf_pred=odf_pred_np)

                #Check --> Plotting
                # if ((i==2)and(j%10==0)):
                #     #Plot target, output novelty curves, target, output spectrograms
                    
                #     plt.figure(figsize=(5,20))
                #     plt.title("Target spectrogram")
                #     plt.imshow((Y_avg[:,x].cpu().numpy()).T, origin='lower')
                #     #plt.show()
                #     filename = "target_spec_"+ str(j) + ".pdf"
                #     plt.savefig(Path(target_path, filename))


                #     plt.figure()
                #     plt.title("Target Novelty curve")
                #     plt.plot(odf_targ_np,label="Training")
                #     #plt.xlabel("Iterations")
                #     #plt.ylabel("Loss")
                #     plt.legend()
                #     #plt.show()
                #     filename = "target_novelty_"+ str(j) + ".pdf"
                #     plt.savefig(Path(target_path, filename))

                #     plt.figure(figsize=(5,20))
                #     plt.title("Predicted spectrogram")
                #     plt.imshow((Y_hat_avg[:,x].cpu().numpy()).T, origin='lower')
                #     #plt.show()
                #     filename = "pred_spec_"+ str(j) + ".pdf"
                #     plt.savefig(Path(target_path, filename))

                #     plt.figure()
                #     plt.title("Predicted Novelty curve")
                #     plt.plot(odf_pred_np,label="Training")
                #     #plt.xlabel("Iterations")
                #     #plt.ylabel("Loss")
                #     plt.legend()
                #     #plt.show()
                #     filename = "pred_novelty_"+ str(j) + ".pdf"
                #     plt.savefig(Path(target_path, filename))



                    



            loss_od = loss_od.to(device)
            loss_od_sf = loss_od_sf.to(device)
            
            mse_loss = criterion2(Y_hat, Y)
            bce_loss = (torch.sum(loss_od)/Y.shape[1])
            bce_sf_loss = (torch.sum(loss_od_sf)/Y.shape[1])
            #ADDED
            precision = (torch.sum(prec)/Y.shape[1])
            recall = (torch.sum(rec)/Y.shape[1])
            f_score = (torch.sum(f_sc)/Y.shape[1])
            
            #print ("VALID_MSE LOSS = ", mse_loss)
            #print("VALID_BCE_LOSS = ", bce_loss)

            #loss = (args.gamma)*(torch.sum(loss_od)) + (1-args.gamma)*criterion2(Y_hat, Y)
            #loss = (args.gamma)*bce_loss + (1-args.gamma)*mse_loss
            #print("VALID_TOTAL LOSS = ", loss)
            #loss = ((args.gamma)/(2.0))*bce_loss + (1-args.gamma)*mse_loss + ((args.gamma)/(2.0))*bce_sf_loss
            loss = (1-args.gamma)*mse_loss + (args.gamma)*bce_sf_loss
        
            #loss = (1-args.gamma)*mse_loss
            #loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
            mse_losses.update(mse_loss.item(), Y.size(1))
            bce_losses.update(bce_loss.item(), Y.size(1))
            bce_sf_losses.update(bce_sf_loss.item(), Y.size(1))
            
            #ADDED
            precision_values.update(float(precision), Y.size(1))
            recall_values.update(float(recall), Y.size(1))
            f_score_values.update(float(f_score), Y.size(1))

            #Checking 
            i = i+1
            
        return losses.avg , mse_losses.avg, bce_losses.avg , bce_sf_losses.avg, precision_values.avg, recall_values.avg, f_score_values.avg


def get_statistics(args, dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        model.STFT(n_fft=args.nfft, n_hop=args.nhop),
        model.Spectrogram(mono=False)
    )

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_chunks = True
    #dataset_scaler.seq_duration = args.seq_dur
    dataset_scaler.seq_duration = 0.0
    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])
        #print("HELLO", np.squeeze(X).shape)
        p = np.squeeze(X)
        scaler.partial_fit(np.concatenate((p[:,0],p[:,1]) )) #CHANGED!!

    # set inital input scaler values
    std = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    return scaler.mean_, std


def spect_to_logmel_spect(n_fft, n_mels, sr, mag_spect, device):
    '''
    Function takes magnitude spectrogram input (in the format of torch tensor) and returns
    log mel spectrogram
    librosa.filters.mel generates conversion matrix, assuming that spectrogram covers range 
    [0, fs/2]. 
    
    
    Parameters:-
    n_fft--> FFT size
    n_mels --> Number of bins in mel spectrogram
    sr --> sampling rate
    
    Input:-
    mag_spect --> Magnitude spectrogram. Shape of input is [time, batch_size, n_channels, fft_bins]
    '''    
    in_shape = np.array(mag_spect.shape)
    sp_ht = in_shape[3]
    out_shape = in_shape
    out_shape[3] = n_mels
    Y_hat_mel = torch.zeros(list(out_shape))
    batch_size = in_shape[1]
    n_channels = 2
    
    #Define filter matrix to convert to mel 
    filt = librosa.filters.mel(n_fft=n_fft, n_mels = n_mels, sr = sr)
    filt_bl = filt[:,:sp_ht]          #To account for number of bins, if spectrogram is bandlimited
    filt_torch = torch.from_numpy(filt_bl)
    filt_torch = filt_torch.to(device)
    
    
    for i in range (batch_size):
        for j in range (n_channels):
            temp = mag_spect[:,i,j,:].permute(1,0)
            temp = temp.to(device)
            Y_hat_mel[:,i,j,:] = torch.mm(filt_torch, temp.double()).permute(1,0)

    Y_hat_mel=10*torch.log10(1e-10+Y_hat_mel)
    
    return(Y_hat_mel)

def makechunks(x,duration):
    '''
    Input - Torch tensor of size [time, batch size, height]
    Output - Torch tensor of size [num_chunks, chunk_duration, batch_size, height]
    '''
    y=torch.zeros([x.shape[0]-duration+1, duration, x.shape[1],x.shape[2]])
    for i_frame in range(x.shape[0]-duration+1):
        y[i_frame] = x[i_frame:i_frame+duration]
    y = y.permute(2,0,3,1)    
    return y[:,:,None, :,:]        

def zeropad(x, duration):
    pad_len = int(duration/2)
    add = torch.zeros((1, x.shape[1], x.shape[2]))
    k = x
    for i in range(pad_len):
        k = torch.cat((add,k),0)
        k = torch.cat((k,add),0)

    return k

def spectral_flux(magStft,bands,fs,device):
        '''
        Inputs :-
        magStft --> [num_frames, num_bins], in the torch tensor format
        bands --> [0, fs/2]
        fs --> sampling rate

        Output 
        '''
        #####
        magStft = magStft.reshape(magStft.shape[1], magStft.shape[0])
        #magStft = magStft.cpu()
        #####
        binLo=int(bands[0]*magStft.shape[0]/(fs/2))
        binHi=int(bands[1]*magStft.shape[0]/(fs/2))
        #specFlux=np.array([0])
        specFlux = torch.zeros((magStft.shape[1]))
        specFlux = specFlux.to(device)
        for hop in range(1,magStft.shape[1]):
                diff = magStft[:,hop]-magStft[:,hop-1]
                diff = (diff + abs(diff))/2
                specFlux[hop]=torch.sum(diff)
        specFlux/=max(specFlux)
        #specFlux = np.atleast_2d(specFlux).T
        #specFlux = torch.from_numpy(specFlux)
        specFlux = specFlux[:,None]

        return specFlux
 
def spectral_flux_vectorised_torch(magStft, device):
        '''
        magStft --> [num_frames, num_bins], in the torch tensor format
        '''
        #magStft = magStft.reshape(magStft.shape[1], magStft.shape[0])
        magStft = magStft.transpose(1,0)
        magStft = 10*torch.log10(1e-10 + magStft)
        diff = magStft[:,1:] - magStft[:,:-1]
        diff=(diff+abs(diff))/2
        specFlux = torch.zeros((magStft.shape[1]))
        specFlux = specFlux.to(device)
        temp = torch.sum(diff, 0)
        temp = temp.to(device)
        specFlux[1:] = temp
        m = max(specFlux)
        #specFlux/=max(specFlux)
        specFlux = specFlux/m
        specFlux = specFlux[:,None]
        return specFlux

def peakPicker(data, threshold):
	peaks=np.array([],dtype='int')
	for ind in range(1,len(data)-1):
		if ((data[ind+1] < data[ind] > data[ind-1]) & (data[ind]>threshold)):
			peaks=np.append(peaks,ind)
	return peaks


def eval_novelty(odf_targ,odf_pred):
	#pick peaks
	#input shape at his point should be [255,1]
	odf_targ_binary=np.zeros(len(odf_targ))
	odf_targ_binary[peakPicker(odf_targ,0.3)]=1.

	odf_pred_binary=np.zeros(len(odf_pred))
	odf_pred_binary[peakPicker(odf_pred,0.3)]=1.

	peakLocsOut=np.where(odf_pred_binary==1.0)[0]
	peakLocsGt=np.where(odf_targ_binary==1.0)[0]

	#evaluate novelty
	nPositives=len(peakLocsGt)
	nTP=0
	tolerance=4
	for i_peak in range(len(peakLocsOut)):
		while(len(peakLocsGt) != 0):
			if abs(peakLocsOut[i_peak] - peakLocsGt[0]) <= int(tolerance/2):
				peakLocsGt=np.delete(peakLocsGt,0)
				nTP+=1
				break
			elif peakLocsOut[i_peak] < peakLocsGt[0]:
				break
			else:
				peakLocsGt=np.delete(peakLocsGt,0)

	nFP=len(peakLocsOut)-nTP

	precision=nTP/(nTP+nFP)
	recall=nTP/nPositives
	f_sc=2*precision*recall/(0.00001+precision+recall)
	return precision, recall, f_sc







def main():
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')

    # which target do we want to train?
    parser.add_argument('--target', type=str, default='tabla',
                        help='target source (will be passed to the dataset)')
    
    # Dataset paramaters
    parser.add_argument('--dataset', type=str, default="aligned",
                        choices=[
                            'musdb', 'aligned', 'sourcefolder',
                            'trackfolder_var', 'trackfolder_fix'
                        ],
                        help='Name of the dataset.')
    parser.add_argument('--root', type=str, help='root path of dataset', default='../rec_data_final/')
    parser.add_argument('--output', type=str, default="../new_mtl_models/model_tabla_mtl_ourmix_test",
                        help='provide output path base folder name')
    parser.add_argument('--model', type=str, help='Path to checkpoint folder')
    #parser.add_argument('--model', type=str, help='Path to checkpoint folder' , default="../new_mtl_sf_models/model_tabla_mtl_mse_pretrain1")
    #parser.add_argument('--model', type=str, help='Path to checkpoint folder' , default='umxhq')
    parser.add_argument('--onset-model', type=str, help='Path to onset detection model weights' , default="/media/Sharedata/rohit/cnn-onset-det/models/apr4/saved_model_0_80mel-0-16000_1ch_44100.pt")


    
    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--gamma', type=float, default=0.0, 
                        help='weighting of different loss components')
    parser.add_argument('--finetune', type=int, default=1, 
                        help='If true(1), then optimiser states from checkpoint model are reset (required for bce finetuning), false if aim is to resume training from where it was left off')
    parser.add_argument('--onset-thresh', type=float, default=0.3, 
                        help='Threshold above which onset is said to occur')
    parser.add_argument('--binarise', type=int, default=0, 
                        help='If=1(true), then target novelty function is made binary, if=0(false), then left as it is')
    parser.add_argument('--onset-trainable', type=int, default=0,
                        help='If=1(true), then onsetCNN will also get trained in finetuning stage, if=0(false) then kept fixed')
    

    # Model Parameters
    parser.add_argument('--seq-dur', type=float, default=6.0,
                        help='Sequence duration in seconds'
                        'value of <=0.0 will use full/variable length')
    parser.add_argument('--unidirectional', action='store_true', default=False,
                        help='Use unidirectional LSTM instead of bidirectional')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=1024,
                        help='STFT hop size')
    

    
    parser.add_argument('--n-mels', type=int, default=80,
                        help='Number of bins in mel spectrogram')
    
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden size parameter of dense bottleneck layers')
    parser.add_argument('--bandwidth', type=int, default=16000,
                        help='maximum model bandwidth in herz')
    parser.add_argument('--nb-channels', type=int, default=2,
                        help='set number of channels for model (1, 2)')
    parser.add_argument('--nb-workers', type=int, default=4,
                        help='Number of workers for dataloader.')

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    print("Using Torchaudio: ", utils._torchaudio_available())
    dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

    repo_dir = os.path.abspath(os.path.dirname(__file__))
    repo = Repo(repo_dir)
    commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)


    train_dataset, valid_dataset, args = data.load_datasets(parser, args)
    print("TRAIN DATASET", train_dataset)
    print("VALID DATASET", valid_dataset)

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1,
        **dataloader_kwargs
    )

    if args.model:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, train_dataset)

    max_bin = utils.bandwidth_to_max_bin(
        train_dataset.sample_rate, args.nfft, args.bandwidth
    )

    unmix = model_mtl.OpenUnmix_mtl(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_channels=args.nb_channels,
        hidden_size=args.hidden_size,
        n_fft=args.nfft,
        n_hop=args.nhop,
        max_bin=max_bin,
        sample_rate=train_dataset.sample_rate
    ).to(device)
    
    #Read trained onset detection network (Model through which target spectrogram is passed)
    # detect_onset = model.onsetCNN().to(device)
    # detect_onset.load_state_dict(torch.load(args.onset_model, map_location='cuda:0'))
        
    # for child in detect_onset.children():
    #     for param in child.parameters():
    #         param.requires_grad = False


    optimizer = torch.optim.Adam(
        unmix.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if args.model:
        model_path = Path(args.model).expanduser()
        with open(Path(model_path, args.target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint['state_dict'])    
        
        if (args.finetune==0):
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            # train for another epochs_trained
            t = tqdm.trange(
                results['epochs_trained'],
                results['epochs_trained'] + args.epochs + 1,
                disable=args.quiet
            )
            print("PICKUP WHERE LEFT OFF", args.finetune)
            train_losses = results['train_loss_history']
            train_mse_losses = results['train_mse_loss_history']
            train_bce_losses = results['train_bce_loss_history']
            train_bce_sf_losses = results['train_bce_sf_loss_history']
            train_precision_values = results['train_precision_history']
            train_recall_values = results['train_recall_history']
            train_f_score_values = results['train_f_score_history']

            valid_losses = results['valid_loss_history']
            valid_mse_losses = results['valid_mse_loss_history']
            valid_bce_losses = results['valid_bce_loss_history']
            valid_bce_sf_losses = results['valid_bce_sf_loss_history']
            valid_precision_values = results['valid_precision_history']
            valid_recall_values = results['valid_recall_history']
            valid_f_score_values = results['valid_f_score_history']

            
            train_times = results['train_time_history']
            best_epoch = results['best_epoch']
            
            es.best = results['best_loss']
            es.num_bad_epochs = results['num_bad_epochs']
            
        else:           
            t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
            train_losses = []
            train_mse_losses = []
            train_bce_losses = []
            train_bce_sf_losses = []
            #ADDED
            train_precision_values = []
            train_recall_values = []
            train_f_score_values = []            
            
            print("NOT PICKUP WHERE LEFT OFF", args.finetune)
            valid_losses = []
            valid_mse_losses = []
            valid_bce_losses = []
            valid_bce_sf_losses = []
            #ADDED
            valid_precision_values = []
            valid_recall_values = []
            valid_f_score_values = []
            
            
            train_times = []
            best_epoch = 0

        #es.best = results['best_loss']
        #es.num_bad_epochs = results['num_bad_epochs']
    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        train_mse_losses = []
        train_bce_losses = []
        train_bce_sf_losses = []
        #ADDED
        train_precision_values = []
        train_recall_values = []
        train_f_score_values = []
        
        valid_losses = []
        valid_mse_losses = []
        valid_bce_losses = []
        valid_bce_sf_losses = []
        #ADDED
        valid_precision_values = []
        valid_recall_values = []
        valid_f_score_values = []
        

        train_times = []
        best_epoch = 0

        #Testing plotting 
    j = 0
    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss, train_mse_loss, train_bce_loss, train_bce_sf_loss, train_precision, train_recall, train_f_score = train(args, unmix, device, train_sampler, optimizer)
        #train_mse_loss = train(args, unmix, device, train_sampler, optimizer, detect_onset=detect_onset)[1]
        #train_bce_loss = train(args, unmix, device, train_sampler, optimizer, detect_onset=detect_onset)[2]
        
        valid_loss, valid_mse_loss, valid_bce_loss, valid_bce_sf_loss, valid_precision, valid_recall, valid_f_score = valid(args, unmix, device, valid_sampler, j, target_path)
        #valid_mse_loss = valid(args, unmix, device, valid_sampler, detect_onset=detect_onset)[1]
        #valid_bce_loss = valid(args, unmix, device, valid_sampler, detect_onset=detect_onset)[2]
        j = j+1


        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        train_mse_losses.append(train_mse_loss)
        train_bce_losses.append(train_bce_loss)
        train_bce_sf_losses.append(train_bce_sf_loss)
        #ADDED
        train_precision_values.append(train_precision)
        train_recall_values.append(train_recall)
        train_f_score_values.append(train_f_score)
        
        valid_losses.append(valid_loss)
        valid_mse_losses.append(valid_mse_loss)
        valid_bce_losses.append(valid_bce_loss)
        valid_bce_sf_losses.append(valid_bce_sf_loss)
        
        #ADDED
        valid_precision_values.append(valid_precision)
        valid_recall_values.append(valid_recall)
        valid_f_score_values.append(valid_f_score)
        

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

        stop = es.step(valid_loss)
        
        #from matplotlib import pyplot as plt 
        
# =============================================================================
#         plt.figure(figsize=(16,12))
#         plt.subplot(2, 2, 1)
#         plt.title("Training loss")
#         plt.plot(train_losses,label="Training")
#         plt.xlabel("Iterations")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.show()
#         #plt.savefig(Path(target_path, "train_plot.pdf"))
#         
#         plt.figure(figsize=(16,12))
#         plt.subplot(2, 2, 2)
#         plt.title("Validation loss")
#         plt.plot(valid_losses,label="Validation")
#         plt.xlabel("Iterations")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.show()
#         #plt.savefig(Path(target_path, "val_plot.pdf"))
# =============================================================================
         
#        plt.figure(figsize=(16,12))
#        plt.subplot(2, 2, 1)
#        plt.title("Training precision")
#        plt.plot(train_precision_values,label="Training")
#        plt.xlabel("Iterations")
#        plt.ylabel("Loss")
#        plt.legend()
#        plt.show()
#        #plt.savefig(Path(target_path, "train_plot.pdf"))
#        
#        plt.figure(figsize=(16,12))
#        plt.subplot(2, 2, 1)
#        plt.title("Training recall")
#        plt.plot(train_recall_values,label="Training")
#        plt.xlabel("Iterations")
#        plt.ylabel("Loss")
#        plt.legend()
#        plt.show()
#        
#        plt.figure(figsize=(16,12))
#        plt.subplot(2, 2, 1)
#        plt.title("Training f_score")
#        plt.plot(train_f_score_values,label="Training")
#        plt.xlabel("Iterations")
#        plt.ylabel("Loss")
#        plt.legend()
#        plt.show()
#        
#        plt.figure(figsize=(16,12))
#        plt.subplot(2, 2, 1)
#        plt.title("Valid precision")
#        plt.plot(valid_precision_values,label="Validation")
#        plt.xlabel("Iterations")
#        plt.ylabel("Loss")
#        plt.legend()
#        plt.show()
#        #plt.savefig(Path(target_path, "train_plot.pdf"))
#        
#        plt.figure(figsize=(16,12))
#        plt.subplot(2, 2, 1)
#        plt.title("Valid recall")
#        plt.plot(valid_recall_values,label="Validation")
#        plt.xlabel("Iterations")
#        plt.ylabel("Loss")
#        plt.legend()
#        plt.show()
#        
#        plt.figure(figsize=(16,12))
#        plt.subplot(2, 2, 1)
#        plt.title("Valid f_score")
#        plt.plot(valid_f_score_values,label="Validation")
#        plt.xlabel("Iterations")
#        plt.ylabel("Loss")
#        plt.legend()
#        plt.show()



        

        if valid_loss == es.best:
            best_epoch = epoch
        
        utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': unmix.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target
        )
        


        # save params
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'train_mse_loss_history': train_mse_losses,
            'train_bce_loss_history': train_bce_losses,
            'train_bce_sf_loss_history': train_bce_sf_losses,
            'train_precision_history' : train_precision_values,
            'train_recall_history' : train_recall_values,
            'train_f_score_history' :train_f_score_values,
            'valid_loss_history': valid_losses,
            'valid_mse_loss_history': valid_mse_losses,
            'valid_bce_loss_history': valid_bce_losses,
            'valid_bce_sf_loss_history': valid_bce_sf_losses,
            'valid_precision_history' : valid_precision_values,
            'valid_recall_history' : valid_recall_values,
            'valid_f_score_history' : valid_f_score_values,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
            'commit': commit
        }

        with open(Path(target_path,  args.target + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break
    
# =============================================================================
#     plt.figure(figsize=(16,12))
#     plt.subplot(2, 2, 1)
#     plt.title("Training loss")
#     #plt.plot(train_losses,label="Training")
#     plt.plot(train_losses,label="Training")
#     plt.xlabel("Iterations")
#     plt.ylabel("Loss")
#     plt.legend()
#     #plt.show()
#     
#     plt.figure(figsize=(16,12))
#     plt.subplot(2, 2, 2)
#     plt.title("Validation loss")
#     plt.plot(valid_losses,label="Validation")
#     plt.xlabel("Iterations")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.show()
#     plt.savefig(Path(target_path, "train_val_plot.pdf"))
#     #plt.savefig(Path(target_path, "train_plot.pdf"))
# =============================================================================
    
    print("TRAINING DONE!!")
    
    plt.figure()
    plt.title("Training loss")
    plt.plot(train_losses,label="Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(target_path, "train_plot.pdf"))
    
    plt.figure()
    plt.title("Validation loss")
    plt.plot(valid_losses,label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(target_path, "val_plot.pdf"))
    
    plt.figure()
    plt.title("Training BCE MTL loss")
    plt.plot(train_bce_losses,label="Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(target_path, "train_bce_plot.pdf"))

    plt.figure()
    plt.title("Training BCE SF loss")
    plt.plot(train_bce_sf_losses,label="Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(target_path, "train_bce_sf_plot.pdf"))    

    plt.figure()
    plt.title("Validation BCE MTL loss")
    plt.plot(valid_bce_losses,label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(target_path, "val_bce_plot.pdf"))

    plt.figure()
    plt.title("Validation BCE SF loss")
    plt.plot(valid_bce_sf_losses,label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(target_path, "val_bce_sf_plot.pdf"))
    
    plt.figure()
    plt.title("Training MSE loss")
    plt.plot(train_mse_losses,label="Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(target_path, "train_mse_plot.pdf"))
    
    plt.figure()
    plt.title("Validation MSE loss")
    plt.plot(valid_mse_losses,label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(target_path, "val_mse_plot.pdf"))
    
    
    
    plt.figure(figsize=(16,12))
    plt.subplot(2, 2, 1)
    plt.title("Training precision")
    plt.plot(train_precision_values,label="Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(Path(target_path, "train_precision_plot.pdf"))
    
    plt.figure(figsize=(16,12))
    plt.subplot(2, 2, 1)
    plt.title("Training recall")
    plt.plot(train_recall_values,label="Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(Path(target_path, "train_recall_plot.pdf"))
    
    plt.figure(figsize=(16,12))
    plt.subplot(2, 2, 1)
    plt.title("Training f_score")
    plt.plot(train_f_score_values,label="Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(Path(target_path, "train_f_score_plot.pdf"))
    
    plt.figure(figsize=(16,12))
    plt.subplot(2, 2, 1)
    plt.title("Valid precision")
    plt.plot(valid_precision_values,label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(Path(target_path, "valid_precision_plot.pdf"))
    
    plt.figure(figsize=(16,12))
    plt.subplot(2, 2, 1)
    plt.title("Valid recall")
    plt.plot(valid_recall_values,label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(Path(target_path, "valid_recall_plot.pdf"))
    
    plt.figure(figsize=(16,12))
    plt.subplot(2, 2, 1)
    plt.title("Valid f_score")
    plt.plot(valid_f_score_values,label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(Path(target_path, "valid_f_score_plot.pdf"))

    
    
    
    
    

if __name__ == "__main__":
    main()
