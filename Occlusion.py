import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Bio import SeqIO
import tensorflow as tf
import subprocess
from utils import to_binary, zero_padding
from Callbacks import coef_det_k

import argparse 

ALANINE = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # sets global vareable ALANINE as a one hot reprecentation of alanine
PADDING = 2000 # sets sequence length 

BLUE        = [0.0, 0.0, 1.0, 1.0]
LIGHTBLUE   = [0.4, 0.7, 1.0, 1.0]
DARKYELLOW  = [0.8, 0.8, 0.0, 1.0]
LIGHTYELLOW = [1.0, 1.0, 0.4, 1.0]
GREY        = [0.9, 0.9, 0.9, 0.5]

COLORMAP = ListedColormap([BLUE, LIGHTBLUE, GREY, LIGHTYELLOW, DARKYELLOW])

parser = argparse.ArgumentParser(description='''
    Take a pdb file of as input. \n
    From this a distance matrix (between beta carbons) is computed
    and saved to file. The numbering (index and columns) reflects
    the amino acids in the structure, where some are often missing.''')

parser.add_argument('--infolder', metavar='', type=str,
                   help='A folder with pdb files containing a protein structure or a tsv file containing all distances.')
parser.add_argument('--outfolder', metavar='', default='./', type=str,
                   help='Folder to which output files should be written.')
parser.add_argument('--imgfolder', metavar='', default = './', type=str,
                   help = 'Folder to save all Plots to')
parser.add_argument('--seq_file', default= 'cleaned_topts.fasta', type = str,
                   help='File with sequence data')
parser.add_argument('--model', type = str,
                   help= 'model to evaluate with occlusion data')

args = parser.parse_args()

assert os.path.isdir(args.infolder), '{} is not a directory'.format(args.infolder)
assert len(os.listdir(args.infolder))>0, '{} does not contain any files'.format(args.infolder)
assert os.path.isfile(args.seq_file), '{} is not a file'.format(args.seq_file)
print(args.seq_file[-5:])
assert args.seq_file[-5:] == 'fasta', 'sequens data must have the format .fasta'
assert os.path.isfile(args.model), '{} is not a file'.format(args.model)
assert args.model[:-2] != 'h5', 'The model needs to have format h5'

if(os.listdir(args.infolder)[0][-3:] is 'pdb' and ~os.path.isdir(args.outfolder)):
    os.mkdir('{}_dist_mats'.format(args.infolder))
    args.outfolder = '{}_dist_mats'.format(args.infolder)
if(os.listdir(args.infolder)[0][-3:]=='pdb'):
    in_ = args.infolder
    out_ = args.outfolder
    for file in os.listdir(args.infolder):
        subprocess.run('python dist_matrix.py --infile {} --outfolder {}'.format(os.path.join(in_,file), out_), shell=True)


    
def load_sequens_data(fname_seq):
    seq_dict = {'id':[], 'ogt':[], 'seq':[], 'seq_bin':[]}
    for rec in SeqIO.parse(fname_seq,'fasta'):
        seq_dict['id'].append(rec.id)
        seq_dict['prop'].append(float(rec.description.split()[-1]))
        seq_dict['seq'].append(rec.seq)
        seq_dict['seq_bin'].append(to_binary(rec.seq))
    return seq_dict

def occlusion_seqs(seq, df, radius, window):
    global ALANINE #Initiates global variable inside local scope
    global PADDING
    
    size_seq = len(seq)
    start = int(df.values[0,0] - 1) # -1 to get from numeric index to data index
    end = int(df.values[-1,0])
    m = np.zeros([size_seq, size_seq], dtype=bool)
    m[start:end,start:end] = df.values[:,1:]<radius
    mod_list_seq_3D = []
    for i in range(m.shape[0]):
        tmp = np.zeros([seq.shape[0],seq.shape[1]])
        tmp[m[i,:],:] = ALANINE # Setting all amino acids within radious to Alanin
        tmp[~m[i,:],:] = seq[~m[i,:],:]
        tmp = zero_padding(tmp, PADDING) #pads sequence to constant length of PADDING to fit model
        mod_list_seq_3D.append(tmp)
    
    mod_list_seq_1D = []
    for i in range(size_seq - window):
        tmp = np.zeros([seq.shape[0],seq.shape[1]])
        logic = np.zeros(seq.shape[0], dtype = bool)
        logic[i:i+window] = True
        tmp[logic,:] = ALANINE
        tmp[~logic,:] = seq[~logic,:]
        tmp = zero_padding(tmp, PADDING)
        mod_list_seq_1D.append(tmp)
        
    return (mod_list_seq_1D, mod_list_seq_3D, m)

def load_model(fname, v = 0):
    model = tf.keras.models.load_model(fname, custom_objects={'coef_det_k': coef_det_k})     
    if v == 1:
        model.summary()
    return model

def plot_occlusion(arr_1D, arr_3D, m, real_ref, predict_ref, directory, fname):
    
    global LIGHTYELLOW, DARKYELLOW, LIGHTBLUE, BLUE, GREY
    
    x_range = np.arange(arr_1D.size)
    ref = np.ones(arr_1D.size)*real_ref
    
    ref_arr = np.ones([arr_3D.size,4])* GREY
    ref_arr[arr_3D > (predict_ref[0] + 0.5),:]  = LIGHTYELLOW 
    ref_arr[arr_3D > (predict_ref[0] + 1), :]   = DARKYELLOW
    ref_arr[arr_3D < (predict_ref[0] - 0.5), :] = LIGHTBLUE
    ref_arr[arr_3D < (predict_ref[0] - 1), :]   = BLUE
    
    fig, axs  = plt.subplots(2,1,sharex = True, gridspec_kw = {'height_ratios': [1,8]})
    (ax1, ax2) = axs
    ax1.imshow(arr_1D.reshape([1,arr_1D.size]), aspect="auto", cmap= COLORMAP)
    #ax1.plot(x_range, ref, color= 'red')
    ax1.set_title('Occlusion along the primary structure')
    plt.setp(ax1.get_yticklabels(), visible=False)
    for i, temp in enumerate(arr_3D):
        h_ofset = arr_3D.size - i
        x_cord = np.nonzero(m[i,:])
        y_cord = np.ones_like(x_cord)*h_ofset
        ax2.scatter(x_cord,y_cord, color = ref_arr[i], s=1)
    ax2.set_title('Occlusion along the second and tertiary structure') 
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=COLORMAP), ax=axs, ticks = [0, 0.25, 0.5, 0.75 ,1])
    cbar.ax.set_yticklabels(['Destabilizing','','Neutral','', 'Stabelizing'])
    tic_l = ax2.get_yticklabels()
    print(list(reversed(tic_l)))
    ax2.set_yticklabels(list(reversed(tic_l)))
    plt.savefig(os.path.join(directory,fname))
        
def main():
    radius = 6 # 6Å
    window = 3 # 3 aminoascids 
    
    seq_dict = load_sequens_data(args.seq_file)
    model = load_model(args.model, v = 0)
    
    for seq in os.listdir(args.outfolder):
        try:
            id_ = seq.split('_')[0]
            sequence = seq_dict['seq_bin'][seq_dict['id'].index(id_)]
            prop_val = seq_dict['prop'][seq_dict['id'].index(id_)]
        except ValueError:
            print(seq[:-4] + ' is not in sequence data')
        original_seq = zero_padding(sequence,PADDING)
        predict_prop_val = model.predict(original_seq.reshape([1,original_seq.shape[0], original_seq.shape[1]]))[0]
        df = pd.read_csv(seq, sep = '\t')
        mod_list_seq_1D, mod_list_seq_3D, m = occlusion_seqs(sequence, df, radius, window)
        
        prediction_arr_1D = np.zeros(len(mod_list_seq_1D)) # Predicting on all the 1D occluded sequences
        for i, mod_seq in enumerate(mod_list_seq_1D):
            prediction_arr_1D[i] = model.predict(mod_seq.reshape([1,mod_seq.shape[0], mod_seq.shape[1]]))[0]
            
        prediction_arr_3D = np.zeros([len(mod_list_seq_3D)]) # Predicting on all the 3D occluded sequences
        for i, mod_seq in enumerate(mod_list_seq_3D):
            prediction_arr_3D[i] = model.predict(mod_seq.reshape([1,mod_seq.shape[0], mod_seq.shape[1]]))[0]
        amax = np.argmax(prediction_arr_3D)
        print(np.nonzero(m[amax,:]))
        plot_occlusion(prediction_arr_1D, prediction_arr_3D, m, prop, predict_prop_val, args.imgfolder, fname = id_)    
        
        
    
    
if __name__ == '__main__':
    main()
