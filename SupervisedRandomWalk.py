# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:47:32 2022

@author: gulrch
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 16:38:19 2021

@author: gulrch
"""

from graph import DataGenerator as dg
from graph import dFrequencies as fr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from data_utilities import data_utility
import statistics as st
import math
import json
import csv
import networkx as nx
import random

def load(fname, size):
    G = nx.DiGraph()
    d = json.load(open(fname))
    G.add_nodes_from(d['nodes'])
    
    for edge in d['edges']:
        if not G.has_edge(edge[0], edge[1]):
            G.add_edge(edge[0], edge[1])
            G.edges[edge[0], edge[1]]['weight'] = edge[2]
            
 
    return G
def get_random_gender_age(G):
    #Randomly choose gender
    total = G['STR']['M']['weight']+G['STR']['F']['weight']
    gender = (['M']*(int(G['STR']['M']['weight']/total*100)+1))+(['F']*(int(G['STR']['F']['weight']/total*100)+1))
    rand_gender = random.choice(gender)
    
    sum_of_succ = G['STR'][rand_gender]['weight']
    age_group = []
    for succ in G.successors(rand_gender):
        age_group+=[succ]*int((G[rand_gender][succ]['weight']/sum_of_succ)*100)
    rand_age_group = random.choice(age_group)
    
    return rand_gender, rand_age_group





#Find sequences using random weights   
def find_weighted_random_path(G, start, end, path, max_):
    
    # Append to the path
    if not path[-1] == start:
        path.append(start)
    
    # If the end has been reached, or the length about to exceed, return
    if start == end or len(path) == max_:
        return path
    #generate list
    next_diseases = []
    gender = path[0]
    age_group = path[1]
    
    for succ in G.successors(start):
        
        if gender in G[start][succ]['weight']:
            if age_group in G[start][succ]['weight'][gender]:
                next_diseases+=[succ]*G[start][succ]['weight'][gender][age_group]
    rand = start
    # Randomly select the next neighbor
    if next_diseases:
        rand = random.choice(next_diseases)
    else:
        #if no patient of a gender and age range than revert back to one step
        print("Start: "+str(start)+" succ: "+str(succ)+" gender: "+gender+" group: "+age_group)
        return path
    
    #traverse it
    find_weighted_random_path(G, rand, end, path, max_)


def plotClusterHistogram(frequencies, path, title, yLabel, ylog):
    #Plot histogram of frequent diseases in a cluster
    
    xlabels = frequencies.keys()
    width = 0.5
    pos1 = np.arange(len(xlabels))
    
    # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos1)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("ICD-10")
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.bar(pos1, frequencies.values(), width, color='b')
    plt.ylim(min(frequencies.values()),max(frequencies.values()))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x: ,.0f}'))
    plt.yscale('log',basey=10)
    plt.savefig(path+"_histogram.png", bbox_inches = "tight") # save as png
    plt.show()
    
def plotClusterBar(men, women, path, title, yLabel, xlabels):
    
    width = 0.5
    pos1 = np.arange(len(xlabels))
    
    # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos1)
    ax.set_xticklabels(xlabels,rotation = 45, ha="right")
    ax.set_xlabel("Age Groups")
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    
    m = [v for k, v in men.items()]
    f = [v for k, v in women.items()]
    
    plt.bar(range(len(m)), m, width=width, color='slategray')
    plt.bar(range(len(f)), f, bottom=m, width=width, color='khaki')

    #plt.ylim(min(men.values()),max(men.values()))
    #plt.yscale('log',basey=10)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x: ,.0f}'))
    
    ax.legend(labels=['Men', 'Women'])
    plt.savefig(path+"_histogram.png", bbox_inches = "tight") # save as png
    plt.show()

#KLD to measure the disimilarity between two probability distributions
def compute_kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div)

def bc(p,q):
    s=0
    for i in range(0,len(p)):
        s+=math.sqrt(p[i]*q[i])
    return s

#Read patients basic information and cost data
def read_patient_basics(pdata):
    BASE_YEAR = 2018
    DATA_PATH_PATIENTS = '../data/cl_patient_cost_labels/'
    #Read patient data
    with open(DATA_PATH_PATIENTS+"hilmo_patient_cost.json", 'r') as file:
           patients = json.load(file)
    
    #format patient cost data 
    p_data = {}
    p_data['M']={}
    p_data['F']={}
    
    #Transgender
    p_data['T']={}
    
    for p in pdata:
        patient = patients['data'][p]
        birthyear = int(patient['birthyear'])
        label =''
        if int(patient['gender'])==1:
            label='M'
        elif int(patient['gender'])==2:
            label='F'
        elif int(patient['gender'])==3:
            label='T'
        
        age = BASE_YEAR - birthyear
        age_range =0
        
        if age>10 and age<21:
            age_range=1
        elif age>20 and age<31:
            age_range=2
        elif age>30 and age<41:
            age_range=3
        elif age>40 and age<51:
            age_range=4
        elif age>50 and age<61:
            age_range=5
        elif age>60 and age<71:
            age_range=6
        elif age>70 and age<81:
            age_range=7
        elif age>80 and age<91:
            age_range=8
        elif age>90 and age<101:
            age_range=9
        elif age>100 and age<111:
            age_range=10
        elif age>110 and age<121:
            age_range=11
        elif age>120 and age<131:
            age_range=12
        
        if age_range in p_data[label]:
            p_data[label][age_range]+=1
        else:
            p_data[label][age_range]=1
    return p_data

def countSequencesSize(data):
    count={}
    for l in data:
        if len(l) in count:
            count[len(l)]+=1
        else:
            count[len(l)]=1
    return count


#Weighted directed graph is already generated and stored in a text file.
DATA = '../data/data_gen/'
GROUP = 'patients_all_graph_fr_control.txt'    
SIZE = 100000

G = load(DATA+GROUP, SIZE)

max_seq=80
min_seq = 5

start = 'STR'

end = 'T'

data_generated = []

while len(data_generated)<SIZE:
      path = []
      #get random gender and age_group
      
      rand_gender, rand_group = get_random_gender_age(G)
      
      #add values to the path
      path.append(rand_gender)
      path.append(rand_group)
      
      #start is a rand_group
      find_weighted_random_path(G, rand_group, end, path, max_seq)
      #dg.find_weighted_random_controlled_path(G, start, end, path, max_seq, weight_list)
      if path[len(path)-1]=='T' and len(path)>(min_seq+2):
          # decrease the edge weight
          #for indx in range(1,len(path)):
              #weight_list[path[indx-1]+path[indx]]-=1
          del path[-1]
          data_generated.append(path)

with open(DATA+'data_generated_'+str(SIZE)+'_'+str(max_seq)+'_control.json', 'w', encoding ='utf8') as json_file:
    json.dump(data_generated, json_file, ensure_ascii = True)
#Validate the generated data set
#Filter the sequences
filter_data=[]
for d in data_generated:
    filter_data.append(d[2:len(d)])
    
#plot top 10 diseases
freq_all = fr.getFrequencies(filter_data)
descPerc = dict(sorted(freq_all.items(), key=lambda x: x[1], reverse=True)[0:10])
plotClusterHistogram(descPerc,'', 'Prevalent diseases in synthetic data', 'Frequency',False)


#count diseases by age group and gender
#age ranges 
xlabels = ['11-20', '21-30', '31-40', '41-50', '51-60','61-70', '71-80', '81-90', '91-100', '101-110','111-120', '121-130']

men ={}
women= {}
for l in xlabels:
    men[xlabels.index(l)+1]=0
    women[xlabels.index(l)+1]=0
    
for d in data_generated:
    if d[0] =='M':
        men[xlabels.index(d[1])+1] += 1
    else:
        women[xlabels.index(d[1])+1] += 1
    
men = {key: value for key, value in sorted(men.items())}
women ={key: value for key, value in sorted(women.items())}

plotClusterBar(men, women, 'gender', 'Patients by age group and gender (Synthetic)', 'Frequencies', xlabels)

#Find the disimilarities between two probability distributions
#values near zero show that both distributions are very similar, close 1 means disimilar distribution
#p is the synthetic dataset distribution
p ={key:value/sum(freq_all.values()) for key, value in sorted(freq_all.items())}



LABLE = 'random_100000_filter_5_RUVWXYZ'
DATA_PATH = '../data/hilmo_tdata/'+LABLE+'.json'
#preprocess the data
pdata, plist, data = data_utility.read_hilmo_random_json(DATA_PATH)
freq_real = fr.getFrequencies(data)
age_range_real = read_patient_basics(pdata)

men_r = {key: value for key, value in sorted(age_range_real['M'].items())}
women_r ={key: value for key, value in sorted(age_range_real['F'].items())}
#find missing keys
missing_keys = {k: 0 for k in women_r if k not in men_r}
men_r.update(missing_keys)
plotClusterBar(men_r, women_r, 'gender', 'Patients by age group and gender (real)', 'Frequencies', xlabels)

#q is the original dataset distribution
q ={key:value/sum(freq_real.values()) for key, value in sorted(freq_real.items())}

#plot top 10 diseases
descPerc = dict(sorted(freq_real.items(), key=lambda x: x[1], reverse=True)[0:10])
plotClusterHistogram(descPerc,'', 'Prevalent diseases in real data', 'Frequency',False)


# more statistics
print('############### Simple Statistics : Real vs Synthetic ################')
print('############### Overall Frequency histograms ################')
p ={key:value/sum(freq_all.values()) for key, value in sorted(freq_all.items())}
q ={key:value/sum(freq_real.values()) for key, value in sorted(freq_real.items())}

#using Batarchya coeficient

sub=[]
p1 =[]
q1=[]
for k in p:
    if k in q:    
        sub.append(abs(p[k]-q[k]))
        p1.append(p[k])
        q1.append(q[k])

print("KLD: "+str(compute_kl_divergence(np.array(p1),np.array(q1))))
print("Overlap BC:"+ str(int(bc(p1,q1)*100)))
print("Average: "+str(sum(sub)/len(sub)))
print("STD: "+str(st.stdev(sub)))
print("Max: "+ str(max(sub)))
print("Min: "+str(min(sub)))


print('############### Men Age-range histograms ################')
m_age_sy= {key:value/sum(men.values()) for key, value in sorted(men.items())}
f_age_sy= {key:value/sum(women.values()) for key, value in sorted(women.items())}

m_age_r= {key:value/sum(men_r.values()) for key, value in sorted(men_r.items())}
f_age_r= {key:value/sum(women_r.values()) for key, value in sorted(women_r.items())}


sub_men = []
men_real = []
men_syn =[]
sub_f = []
f_real = []
f_syn =[]
for k in m_age_r:
    if k in m_age_sy:    
        sub_men.append(abs(m_age_sy[k]-m_age_r[k]))
        if m_age_sy[k]>0 and m_age_r[k]>0:
            men_real.append(m_age_r[k])
            men_syn.append(m_age_sy[k])
        sub_f.append(abs(f_age_sy[k]-m_age_r[k]))
        if f_age_sy[k]>0 and f_age_r[k]>0:
            f_real.append(f_age_r[k])
            f_syn.append(f_age_sy[k])



print("KLD: "+str(compute_kl_divergence(np.array(men_real),np.array(men_syn))))
print("Overlap BC:"+ str(int(bc(men_real,men_syn)*100)))
print("Average: "+str(sum(sub_men)/len(sub_men)))
print("STD: "+str(st.stdev(sub_men)))
print("Max: "+ str(max(sub_men)))
print("Min: "+str(min(sub_men)))

print('############### Female Age-range histograms ################')

print("KLD: "+str(compute_kl_divergence(np.array(f_real),np.array(f_syn))))
print("Overlap BC:"+ str(int(bc(f_real,f_syn)*100)))
print("Average: "+str(sum(sub_f)/len(sub_f)))
print("STD: "+str(st.stdev(sub_f)))
print("Max: "+ str(max(sub_f)))
print("Min: "+str(min(sub_f)))



print('############### Data length histograms ################')
count_syn=countSequencesSize(filter_data)
count_real=countSequencesSize(data)
#find missing keys and remove them
missing_keys = {k:0 for k in count_syn if k not in count_real}
for k in missing_keys:
    del count_syn[k]
missing_keys = {k:0 for k in count_real if k not in count_syn}
for k in missing_keys:
    del count_real[k]

#plot top 10 lengthy disease codes
descPerc = dict(sorted(count_real.items(), key=lambda x: x[1], reverse=True)[0:10])
plotClusterHistogram(descPerc,'', 'Sequence length in real data', 'Frequency',False)

descPerc = dict(sorted(count_syn.items(), key=lambda x: x[1], reverse=True)[0:10])
plotClusterHistogram(descPerc,'', 'Sequence length in synthetic data', 'Frequency',False)
fr_syn= {key:value/sum(count_syn.values()) for key, value in sorted(count_syn.items())}
fr_real= {key:value/sum(count_real.values()) for key, value in sorted(count_real.items())}



fr_sub= [abs(fr_real[key]-fr_syn[key]) for key in fr_real]
fr_syn = list(fr_syn.values())
fr_real= list(fr_real.values())

    
print("KLD: "+str(compute_kl_divergence(np.array(fr_real),np.array(fr_syn))))
print("Overlap BC:"+ str(int(bc(fr_real,fr_syn)*100)))
print("Average: "+str(sum(fr_sub)/len(fr_sub)))
print("STD: "+str(st.stdev(fr_sub)))
print("Max: "+ str(max(fr_sub)))
print("Min: "+str(min(fr_sub)))



DATA_PATH = '../data/hilmo_tdata/'
#write statistics in cs file
with open(DATA_PATH+'diagnosis_frequencies_control.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Diagnosis','Real','Synthetic', ])
       
    for key in freq_real:
        if key in freq_all:
            writer.writerow([key, str(freq_real[key]),  str(freq_all[key])])



