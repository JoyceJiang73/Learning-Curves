#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import torchtext
import datasets
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
import os


# In[2]:


from datasets import load_dataset
dataset = load_dataset("SetFit/emotion")
print(dataset)
print(dataset['train'][88]['text'])


# In[3]:


from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch
global_vectors = GloVe(name='6B', dim=300)


# In[4]:


embedding_list = []
for text in dataset['train']['text']:
    embedding_list.append(global_vectors.get_vecs_by_tokens(tokenizer(text), lower_case_backup=True))


# In[5]:


from torch.nn.utils.rnn import pad_sequence
inputs = pad_sequence(embedding_list,batch_first=True)


# In[6]:


#np.save("npy/padded_nlp_inputs",inputs)
#load keypoint input array from local 
inputs= np.load("npy/padded_nlp_inputs.npy", allow_pickle = True)
inputs.shape


# In[7]:


import itertools
gestures = []
for i in Counter(dataset['train']['label']).keys():
    gestures.append(i)
pairs = list(itertools.combinations(gestures, 2))
len(pairs)


# In[8]:


inputs.shape


# # RNN-LSTM

# In[9]:


#define device to run model on either cpu or gpu (cuda)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device


# In[10]:


class LSTM(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.lstm = nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.bn = nn.BatchNorm1d(inputs.shape[1])
        self.lstm_array = []
        
    def forward(self,inputs):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_num, inputs.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_num, inputs.size(0), self.hidden_dim).requires_grad_()
        x = self.bn(inputs)
        lstm_out,(hn,cn) = self.lstm(x)
        #out = F.leaky_relu(hn)
        self.lstm_array = lstm_out
        out = self.fc(lstm_out[:,-1,:])
        return out

n_hidden = 300
n_joints = 300
n_categories = 2
n_layer = 1
model = LSTM(n_joints,n_hidden,n_categories,n_layer)
model.to(device)


# In[11]:


def test(model, test_dl):
    loss = 0
    correct = 0
    accuracy = 0

    target_true = 0
    predicted_true = 0
    correct_true = 0
    
    for (X, y) in test_dl:
        output = model(X)
        loss += nn.CrossEntropyLoss()(output, y)
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data).sum()
    
        target_true += torch.sum(y)
        predicted_true += torch.sum(pred)
        correct_true += sum(y*pred == 1)
    
    loss /= len(test_dl.dataset)
    accuracy = 100 * correct/len(test_dl.dataset)
    
    recall = correct_true / target_true
    precision = correct_true / predicted_true
    f1_score = 2 * precision * recall / (precision + recall)

    return accuracy, loss, recall, precision, f1_score

def train(model, lr, num_epochs, train_dl, test_dl, sim_count,epoch_performance):

    opt = torch.optim.Adam(model.parameters(), lr = lr) 
    
    logit_allEpoch = []
    SVC_allEpoch = []
    KNN_allEpoch = []
    FCN_allEpoch = []
    
    for epoch in range(1, num_epochs + 1):
        lstm_arrays = []
        ys = []
        for X, y in train_dl:
            opt.zero_grad()
            prediction = model(X)
            lstm_arrays.append(model.lstm_array)
            ys.append(y)
            loss = nn.CrossEntropyLoss()(prediction, y)  
            loss.backward() 
            opt.step()
            
        test_accuracy, loss, recall, precision, f1_score = test(model, test_dl)
        print(f"Test accuracy at epoch {epoch}: {test_accuracy:.4f}")
        
        #generate performance df with all epoches
        epoch_performance['epoch_accuracy'].append(test_accuracy.item())
        epoch_performance['epoch_loss'].append(loss.item())
        epoch_performance['epoch_recall'].append(recall.item())
        epoch_performance['epoch_precision'].append(precision.item())
        epoch_performance['epoch_f1'].append(f1_score.item())

        #generate processing curve for each epoch
        cat_lstm = torch.cat((lstm_arrays))
        cat_ys = torch.cat((ys))
        #print(cat_lstm.shape)
        #print(cat_ys.shape)
        cpu_lstm = torch.tensor(cat_lstm, device = 'cpu')
        cpu_ys = torch.tensor(cat_ys, device = 'cpu')
        np.save("npy/lstm_array/simulation/lstm/"+str(a)+"&"+str(b)+"_lstm_epoch"+str(epoch),cpu_lstm)
        np.save("npy/lstm_array/simulation/ys/"+str(a)+"&"+str(b)+"_ys_epoch"+str(epoch),cpu_ys)
        #cpu_lstm= np.load("npy/lstm_array/simulation/lstm/"+str(a)+"&"+str(b)+"_lstm.npy", allow_pickle = True)
        #cpu_ys= np.load("npy/lstm_array/simulation/ys/"+str(a)+"&"+str(b)+"_ys.npy", allow_pickle = True)
        
        # generate_lstm_graph_logit(cpu_lstm, cpu_ys, logit_allEpoch)
        # generate_lstm_graph_SVC(cpu_lstm, cpu_ys, SVC_allEpoch)
        # generate_lstm_graph_KNN(cpu_lstm, cpu_ys, KNN_allEpoch)
        # generate_lstm_graph_FCN(cpu_lstm, cpu_ys, FCN_allEpoch)
        
    # metrics_epoch_df = pd.DataFrame(np.vstack(logit_allEpoch))
    # metrics_epoch_df = metrics_epoch_df.T
    # metrics_epoch_df.to_csv('epoch_dfs/sim'+str(sim_count)+'/processing curve sim_Logit_Class'+str(a)+"-Class" + str(b)+'.csv')

    # metrics_epoch_df = pd.DataFrame(np.vstack(SVC_allEpoch))
    # metrics_epoch_df = metrics_epoch_df.T
    # metrics_epoch_df.to_csv('epoch_dfs/sim'+str(sim_count)+'/processing curve sim_SVC_Class'+str(a)+"-Class" + str(b)+'.csv')

    # metrics_epoch_df = pd.DataFrame(np.vstack(KNN_allEpoch))
    # metrics_epoch_df = metrics_epoch_df.T
    # metrics_epoch_df.to_csv('epoch_dfs/sim'+str(sim_count)+'/processing curve sim_KNN_Class'+str(a)+"-Class" + str(b)+'.csv')

    # metrics_epoch_df = pd.DataFrame(np.vstack(FCN_allEpoch))
    # metrics_epoch_df = metrics_epoch_df.T
    # metrics_epoch_df.to_csv('epoch_dfs/sim'+str(sim_count)+'/processing curve sim_FCN_Class'+str(a)+"-Class" + str(b)+'.csv')


# # Data Augmentation - Oversampling

# In[12]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter


# In[13]:


def run_model(inputs, sample):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = LSTM(n_joints,n_hidden,n_categories,n_layer)
    model.to(device)

    features = []
    validation_losses = []
    test_losses = []
    validation_accuracies = []
    test_accuracies = []
    test_recalls = []
    test_precisions = []
    test_F1s=[]
    aug_train_data = []
    
    pad_size = inputs.shape[1]
    emb_size = inputs.shape[2]

    y = sample['class']
    if sum(y)==0:
        print('no presence of this feature')
    else:
        #device = torch.device('cpu')
        X = torch.tensor(inputs, dtype=torch.float).to('cuda')
        y = torch.tensor(y, dtype=torch.long).to('cuda')

        full_dataset=[[X[b], y[b]] for b in range(len(y))]
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        full_train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        X = list(zip(*full_train_dataset))[0]
        y = list(zip(*full_train_dataset))[1]

        #Oversampling 
        reshape_x = torch.tensor((torch.stack(X,dim=0)).squeeze(0)).reshape(len(y), pad_size*emb_size)
        oversample = RandomOverSampler(sampling_strategy='not majority')
        X_over, y_over = oversample.fit_resample(reshape_x.cpu(), (torch.stack(y,dim=0)).squeeze(0).cpu())
        
        #Reshaping back for RNN input
        X_over = torch.tensor(X_over, dtype=torch.float).reshape(len(X_over), pad_size, emb_size)
        y_over = torch.tensor(y_over, dtype=torch.long)
        X = X_over
        y = y_over
        print(len(y_over))
        aug_train_data.append(len(y_over))
        
        #Sent to cuda for faster processing
        X = X.to(device)
        y = y.to(device)

        full_train_dataset=[[X[i], y[i]] for i in range(len(y))]

        train_size = int(0.8 * len(full_train_dataset))
        valid_size = len(full_train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, valid_size])

        batch_size=64
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
      
        #Run the RNN-LSTM model
        model = LSTM(n_joints,n_hidden,n_categories,n_layer)
        model = model.cuda()

        lr = 0.001
        num_epochs = 20

        epoch_performance = {'epoch_accuracy': [],
     'epoch_loss': [],
     'epoch_recall': [],
     'epoch_precision': [],
     'epoch_f1': []}

        train(model, lr, num_epochs, train_dl, valid_dl, sim_count,epoch_performance)
        validation_accuracy, validation_loss, val_recall, val_precision, val_f1_score = test(model, valid_dl)
        test_accuracy, test_loss, test_recall, test_precision, test_f1_score = test(model, test_dl)
        
        epoch_performance_df = pd.DataFrame.from_dict(epoch_performance, orient='index').T
        epoch_performance_df.to_csv("performance_output/simulation/allEpochPerformance"+str(a)+"&"+str(b)+"_sim"+str(sim_count)+".csv")

        validation_accuracies.append(round(validation_accuracy.item(), 4))
        validation_losses.append(round(validation_loss.item(), 4))
        test_accuracies.append(round(test_accuracy.item(),4))
        test_losses.append(round(test_loss.item(),4))
        test_recalls.append(round(test_recall.item(),4))
        test_precisions.append(round(test_precision.item(),4))
        test_F1s.append(round(test_f1_score.item(),4))
        print(f"Final validation accuracy on OpenPose: {validation_accuracy:.4f}")
        print(f"Final test accuracy on OpenPose: {test_accuracy:.4f}")
        
    performance = pd.DataFrame(
        {'aug_train_data': aug_train_data,
     'val_accuracy': validation_accuracies,
     'val_loss': validation_losses,
     'test_accracy': test_accuracies,
     'test_loss': test_losses,
     'test_recall': test_recalls,
     'test_precision': test_precisions,
     'test_f1': test_F1s
        })
    
    performance.to_csv("performance_output/simulation/finalPerformance"+str(a)+"&"+str(b)+"_sim"+str(sim_count)+".csv")
    torch.save(model.state_dict(), os.path.join("model/simulation/", str(a)+"&"+str(b)+'_simulation.pt'))


# # Mapping Learning Curve - Pair-wise Model

# In[14]:

from sklearnex import patch_sklearn
#patch_sklearn()
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neural_network import MLPClassifier
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from daal4py import daalinit
#daalinit(1)

# In[15]:


def generate_lstm_graph_logit(cpu_lstm, cpu_ys, metrics_simulations):
    data_size = int(cpu_lstm.shape[0])
    metric_timesteps = []
    for i in range(inputs.shape[1]):
        model = LogisticRegression()
        model.fit(cpu_lstm[-data_size:, i, : ], cpu_ys[-data_size:])
        train_preds = model.predict(cpu_lstm[-data_size:, i, : ]) 
        accuracy = accuracy_score(cpu_ys[-data_size:], train_preds)
        metric = accuracy
        metric_timesteps.append(metric)
        #print('logit')
        #print(i, accuracy)
    metrics_simulations.append(metric_timesteps)


# In[16]:


def generate_lstm_graph_SVC(cpu_lstm, cpu_ys, metrics_simulations):
    data_size = int(cpu_lstm.shape[0])
    metric_timesteps = []
    for i in range(inputs.shape[1]):
        model = SVC()
        model.fit(cpu_lstm[-data_size:, i, : ], cpu_ys[-data_size:])
        train_preds = model.predict(cpu_lstm[-data_size:, i, : ])
        accuracy = accuracy_score(cpu_ys[-data_size:], train_preds)
        metric = accuracy
        metric_timesteps.append(metric)
        #print('SVC')
        #print(i, accuracy)
    metrics_simulations.append(metric_timesteps)


# In[17]:


def generate_lstm_graph_KNN(cpu_lstm, cpu_ys, metrics_simulations):
    data_size = int(cpu_lstm.shape[0])
    metric_timesteps = []
    for i in range(inputs.shape[1]):
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(cpu_lstm[-data_size:, i, : ], cpu_ys[-data_size:])
        train_preds = model.predict(cpu_lstm[-data_size:, i, : ])
        accuracy = accuracy_score(cpu_ys[-data_size:], train_preds)
        metric = accuracy
        metric_timesteps.append(metric)
        #print('KNN')
        #print(i, accuracy)
    metrics_simulations.append(metric_timesteps)


# In[18]:


def generate_lstm_graph_FCN(cpu_lstm, cpu_ys, metrics_simulations):
    data_size = int(cpu_lstm.shape[0])
    metric_timesteps = []
    for i in range(inputs.shape[1]):
        model = MLPClassifier(hidden_layer_sizes=(10,))
        model.fit(cpu_lstm[-data_size:, i, : ], cpu_ys[-data_size:])
        accuracy = model.score(cpu_lstm[-data_size:, i, : ], cpu_ys[-data_size:])
        metric = accuracy
        metric_timesteps.append(metric)
        #print('FCN')
        #print(i, accuracy)
    metrics_simulations.append(metric_timesteps)


# ## Export simulation dfs and graphs 

# In[19]:


df = pd.DataFrame(dataset['train']['label'], columns =['class'], dtype = float) 

import sys
input1=int(sys.argv[1])
input2=int(sys.argv[2])

for (a, b) in [pairs[input1]]:
    sim = [(a,b)]*1
    
#for (a, b) in pairs[:1]:
    logit_simulations = []
    SVC_simulations = []
    KNN_simulations = []
    FCN_simulations = []
    
    sim_count = input2
    for (a, b) in sim:
        print(a, b)

        df['class_selected'] = [1 if x==a or x==b else 0 for x in df['class']]
    
        new_inputs = []
        for i in range(len(df['class_selected'])):
            if df['class_selected'][i] ==1:
                new_inputs.append(inputs[i])
        sample = df[(df['class']==a) | (df['class']==b)]
        sample['class'] = sample['class'].replace([a, b], [0, 1])
    
        #pad_frames, valid = padding(new_inputs, 274)
        #sample['valid'] = valid
        #np.save("npy/simulation/input/pose_class"+str(a)+"&class"+str(b)+"_simulation",new_inputs)
        np.save("npy/simulation/pad_input/pad_class"+str(a)+"&class"+str(b)+"_simulation",new_inputs)

        #exclude invalid data points 
        #sample_valid = sample[sample['valid']==1]
        sample_valid = sample.reset_index(drop=True)
    
        emotion= np.load("npy/simulation/pad_input/pad_class"+str(a)+"&class"+str(b)+"_simulation.npy", allow_pickle = True)
        run_model(emotion, sample_valid)
        
        sim_count +=1

#     #Logit
#     metrics_simulations_df = pd.DataFrame(np.vstack(logit_simulations))
#     metrics_simulations_df = metrics_simulations_df.T
#     metrics_simulations_df.to_csv('simulation_dfs/learning curve sim_Logit_Class'+str(a)+"-Class" + str(b)+'.csv')
#     ax = metrics_simulations_df.plot(figsize=(20,10), legend = False)
#     ax.set_title("Simulation Result of 100 Runs(Class"+str(a)+"-Class" + str(b)+") \n Learning Curve of the LSTM Array ", fontsize=30)
#     ax.tick_params(axis='x', which='both', labelsize=25)
#     ax.tick_params(axis='y', which='both', labelsize=25)
#     ax.set_xlabel('LSTM timesteps of the last epoch',fontdict={'fontsize':22})
#     ax.set_ylabel('Logit Accuracy (grey)',fontdict={'fontsize':22}).get_figure().savefig("plot/simulation_100runs/Logit_Class"+str(a)+"-Class" + str(b)+".png")

#     #SVC
#     metrics_simulations_df = pd.DataFrame(np.vstack(SVC_simulations))
#     metrics_simulations_df = metrics_simulations_df.T
#     metrics_simulations_df.to_csv('simulation_dfs/learning curve sim_SVC_Class'+str(a)+"-Class" + str(b)+'.csv')
#     ax = metrics_simulations_df.plot(figsize=(20,10), 
#                  legend = False)
#         #ax.set_title('Learning Curve of the OpenPose LSTM Array \n Increasing representional capacity of visual information', fontsize=30)
#     ax.set_title("Simulation Result of 100 Runs(Class"+str(a)+"-Class" + str(b)+") \n Learning Curve of the LSTM Array ", fontsize=30)
#     ax.tick_params(axis='x', which='both', labelsize=25)
#     ax.tick_params(axis='y', which='both', labelsize=25)
#     ax.set_xlabel('LSTM timesteps of the last epoch',fontdict={'fontsize':22})
#         #ax.set_ylabel('SVM Accuracy (grey)',fontdict={'fontsize':22}).get_figure().savefig("plot/simulation_100runs/FCN/FCN_Class"+str(a)+"-Class" + str(b)+".png")
#     ax.set_ylabel('SVC Accuracy (grey)',fontdict={'fontsize':22}).get_figure().savefig("plot/simulation_100runs/SVC_Class"+str(a)+"-Class" + str(b)+".png")

#     #KNN
#     metrics_simulations_df = pd.DataFrame(np.vstack(KNN_simulations))
#     metrics_simulations_df = metrics_simulations_df.T
#     metrics_simulations_df.to_csv('simulation_dfs/learning curve sim_KNN_Class'+str(a)+"-Class" + str(b)+'.csv')
#     ax = metrics_simulations_df.plot(figsize=(20,10), 
#                  legend = False)
#         #ax.set_title('Learning Curve of the OpenPose LSTM Array \n Increasing representional capacity of visual information', fontsize=30)
#     ax.set_title("Simulation Result of 100 Runs(Class"+str(a)+"-Class" + str(b)+") \n Learning Curve of the LSTM Array ", fontsize=30)
#     ax.tick_params(axis='x', which='both', labelsize=25)
#     ax.tick_params(axis='y', which='both', labelsize=25)
#     ax.set_xlabel('LSTM timesteps of the last epoch',fontdict={'fontsize':22})
#         #ax.set_ylabel('SVM Accuracy (grey)',fontdict={'fontsize':22}).get_figure().savefig("plot/simulation_100runs/FCN/FCN_Class"+str(a)+"-Class" + str(b)+".png")
#     ax.set_ylabel('KNN Accuracy (grey)',fontdict={'fontsize':22}).get_figure().savefig("plot/simulation_100runs/KNN_Class"+str(a)+"-Class" + str(b)+".png")

#     #FCN
#     metrics_simulations_df = pd.DataFrame(np.vstack(FCN_simulations))
#     metrics_simulations_df = metrics_simulations_df.T
#     metrics_simulations_df.to_csv('simulation_dfs/learning curve sim_FCN_Class'+str(a)+"-Class" + str(b)+'.csv')
#     ax = metrics_simulations_df.plot(figsize=(20,10), 
#                  legend = False)
#         #ax.set_title('Learning Curve of the OpenPose LSTM Array \n Increasing representional capacity of visual information', fontsize=30)
#     ax.set_title("Simulation Result of 100 Runs(Class"+str(a)+"-Class" + str(b)+") \n Learning Curve of the LSTM Array ", fontsize=30)
#     ax.tick_params(axis='x', which='both', labelsize=25)
#     ax.tick_params(axis='y', which='both', labelsize=25)
#     ax.set_xlabel('LSTM timesteps of the last epoch',fontdict={'fontsize':22})
#         #ax.set_ylabel('SVM Accuracy (grey)',fontdict={'fontsize':22}).get_figure().savefig("plot/simulation_100runs/FCN/FCN_Class"+str(a)+"-Class" + str(b)+".png")
#     ax.set_ylabel('FCN Accuracy (grey)',fontdict={'fontsize':22}).get_figure().savefig("plot/simulation_100runs/FCN_Class"+str(a)+"-Class" + str(b)+".png")
    


# In[ ]:




