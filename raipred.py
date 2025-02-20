#########################################################################################
# RAIpred is developed for predicting, desigining and scanning rheumatoid arthritis disease causing    #
# peptides using sequence information. It is developed by Prof G. P. S. Raghava's group.#
# Please cite: https://webs.iiitd.edu.in/raghava/raipred/                           #
#########################################################################################
import argparse
import warnings
import subprocess
import pkg_resources
import os
import sys
import numpy as np
import pandas as pd
import math
import itertools
from collections import Counter
from itertools import combinations
import re
import glob
import time
import uuid
from time import sleep
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import pickle
import zipfile
import subprocess

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Please provide following arguments') 

## Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence(s) in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output",type=str, help="Output: File for saving results by default outfile.csv")
#parser.add_argument("-m", "--method",type=int, choices = [1,2], help="Method Type: 1:Main (Disease Vs Random), 2:Alternate (Disease Vs Non-Disease),  by default 1")
parser.add_argument("-j", "--job",type=int, choices = [1,2,3,4], help="Job Type: 1:Predict, 2:Design, 3:Scan, 4: Ensemble Method,  by default 1")
parser.add_argument("-t","--threshold", type=float, help="Threshold: Value between 0 to 1 by default 0.72")
parser.add_argument("-w","--winleng", type=int, choices =range(9, 21), help="Window Length: 8 to 20 (scan mode only), by default 9")
parser.add_argument("-d","--display", type=int, choices = [1,2], help="Display: 1:Only Disease-Causing Peptides, 2: All peptides, by default 1")
args = parser.parse_args()

# Function for generating all possible mutants
def mutants(file1,file2):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    cc = []
    dd = []
    ee = []
    df2 = file2
    df2.columns = ['Name']
    df1 = file1
    df1.columns = ['Seq']
    for k in range(len(df1)):
        cc.append(df1['Seq'][k])
        dd.append('Original_'+'Seq'+str(k+1))
        ee.append(df2['Name'][k])
        for i in range(0,len(df1['Seq'][k])):
            for j in std:
                if df1['Seq'][k][i]!=j:
                    dd.append('Mutant_'+df1['Seq'][k][i]+str(i+1)+j+'_Seq'+str(k+1))
                    cc.append(df1['Seq'][k][:i] + j + df1['Seq'][k][i + 1:])
                    ee.append(df2['Name'][k])
    xx = pd.concat([pd.DataFrame(ee),pd.DataFrame(dd),pd.DataFrame(cc)],axis=1)
    xx.columns = ['Seq_ID','Mutant_ID','Seq']
    return xx
# Function for generating pattern of a given length
def seq_pattern(file1,file2,num):
    df1 = file1
    df1.columns = ['Seq']
    df2 = file2
    df2.columns = ['Name']
    cc = []
    dd = []
    ee = []
    for i in range(len(df1)):
        for j in range(len(df1['Seq'][i])):
            xx = df1['Seq'][i][j:j+num]
            if len(xx) == num:
                cc.append(df2['Name'][i])
                dd.append('Pattern_'+str(j+1)+'_Seq'+str(i+1))
                ee.append(xx)
    df3 = pd.concat([pd.DataFrame(cc),pd.DataFrame(dd),pd.DataFrame(ee)],axis=1)
    df3.columns= ['Seq_ID','Pattern_ID','Seq']
    return df3

# Function to check the sequence
def readseq(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid = []
    seq = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '', ''.join(array[1:]).upper())
        seqid.append('>' + name)
        seq.append(sequence)
    if len(seqid) == 0:
        with open(file, "r") as f:
            data1 = f.readlines()
        for each in data1:
            seq.append(each.strip())
        for i in range(1, len(seq) + 1):
            seqid.append(">Seq_" + str(i))
    df1 = pd.DataFrame(seqid)
    df2 = pd.DataFrame(seq)
    return df1, df2

# Function to check the length of sequences
def lenchk(file1):
    cc = []
    df1 = file1
    df1.columns = [0]
    for i in range(len(df1)):
        cc.append(df1[0][i][:20] if len(df1[0][i]) > 20 else df1[0][i])
    df2 = pd.DataFrame(cc, columns=[0])
    return df2

# Function to extract CeTD feature
def ctd(file):
    attr=pd.read_csv("Data/aa_attr_group.csv", sep="\t")
    filename, file_extension = os.path.splitext(file)
    df1 = pd.read_csv(file, header = None)
    df = pd.DataFrame(df1[0].str.upper())
    n = 0
    stt1 = []
    m = 1
    for i in range(0,len(attr)) :
        st =[]
        stt1.append([])
        for j in range(0,len(df)) :
            stt1[i].append([])
            for k in range(0,len(df[0][j])) :
                while m < 4 :
                    while n < len(attr.iloc[i,m]) :
                        if df[0][j][k] == attr.iloc[i,m][n] :
                            st.append(m)
                            stt1[i][j].append(m)
                        n += 2
                    n = 0
                    m += 1
                m = 1
#####################Composition######################
    f = open("compout_1", 'w')
    sys.stdout = f
    std = [1,2,3]
    print("1,2,3,")
    for p in range (0,len(df)) :
        for ii in range(0,len(stt1)) :
            #for jj in stt1[ii][p]:
            for pp in std :
                count = 0
                for kk in stt1[ii][p] :
                    temp1 = kk
                    if temp1 == pp :
                        count += 1
                    composition = (count/len(stt1[ii][p]))*100
                print("%.2f"%composition, end = ",")
            print("")
    f.truncate()

#################################Transition#############
    tt = []
    tr=[]
    kk =0
    for ii in range(0,len(stt1)) :
        tt = []
        tr.append([])
        for p in range (0,len(df)) :
            tr[ii].append([])
            while kk < len(stt1[ii][p]) :
                if kk+1 <len(stt1[ii][p]):
                #if  stt1[ii][p][kk] < stt1[ii][p][kk+1] or stt1[ii][p][kk] > stt1[ii][p][kk+1]: # condition for adjacent values
                    tt.append(stt1[ii][p][kk])
                    tt.append(stt1[ii][p][kk+1])
                    tr[ii][p].append(stt1[ii][p][kk])
                    tr[ii][p].append(stt1[ii][p][kk+1])

                kk += 1
            kk = 0
    pp = 0
    xx = []
    xxx = []
    for mm in range(0,len(tr)) :
        xx = []
        xxx.append([])
        for nn in range(0,len(tr[mm])):
            xxx[mm].append([])
            while pp < len(tr[mm][nn]) :
                xx .append(tr[mm][nn][pp:pp+2])
                xxx[mm][nn].append(tr[mm][nn][pp:pp+2])
                pp+=2
            pp = 0

    f1 = open("compout_2", 'w')
    sys.stdout = f1
    std1 = [[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]
    print("1->1,1->2,1->3,2->1,2->2,2->3,3->1,3->2,3->3,")
    for rr in range(0,len(df)) :
        for qq in range(0,len(xxx)):
            for tt in std1 :
                count = 0
                for ss in xxx[qq][rr] :
                    temp2 = ss
                    if temp2 == tt :
                        count += 1
                print(count, end = ",")
            print("")
    f1.truncate()

    #################################Distribution#############
    c_11 = []
    c_22 = []
    c_33 = []
    zz = []
    #print("0% 25% 50% 75% 100%")
    for x in range(0,len(stt1)) :
        #c_11.append([])
        c_22.append([])
        #c_33.append([])
        yy_c_1 = []
        yy_c_2 = []
        yy_c_3 = []
        ccc = []

        k = 0
        j = 0
        for y in range(0,len(stt1[x])):
            #c_11[x].append([])
            c_22[x].append([])
            for i in range(1,4) :
                cc = []
                c1 = [index for index,value in enumerate(stt1[x][y]) if value == i]
                c_22[x][y].append(c1)
    cc = []
    for ss in range(0,len(df)):
        for uu in range(0,len(c_22)):
            for mm in range(0,3):
                for ee in range(0,101,25):
                    k = (ee*(len(c_22[uu][ss][mm])))/100
                    cc.append(math.floor(k))
    f2 = open('compout_3', 'w')
    sys.stdout = f2
    print("0% 25% 50% 75% 100%")
    for i in range (0,len(cc),5):
        print(*cc[i:i+5])
    f2.truncate()
    head = []
    header1 = ['CeTD_HB','CeTD_VW','CeTD_PO','CeTD_PZ','CeTD_CH','CeTD_SS','CeTD_SA']
    for i in header1:
        for j in range(1,4):
            head.append(i+str(j))
    df11 = pd.read_csv("compout_1")
    df_1 = df11.iloc[:,:-1]
    zz = pd.DataFrame()
    for i in range(0,len(df_1),7):
        temp_df = pd.DataFrame(pd.concat([df_1.loc[i], df_1.loc[i+1], df_1.loc[i+2], df_1.loc[i+3], df_1.loc[i+4], df_1.loc[i+5], df_1.loc[i+6]], axis=0)).transpose()
        zz = pd.concat([zz, temp_df], ignore_index=True)

        #zz = zz.append(pd.DataFrame(pd.concat([df_1.loc[i],df_1.loc[i+1],df_1.loc[i+2],df_1.loc[i+3],df_1.loc[i+4],df_1.loc[i+5],df_1.loc[i+6]],axis=0)).transpose()).reset_index(drop=True)
    zz.columns = head
        #zz.to_csv(filename+".ctd_comp", index=None, encoding='utf-8')
    head2 = []
    header2 = ['CeTD_11','CeTD_12','CeTD_1-3','CeTD_21','CeTD_22','CeTD_23','CeTD_31','CeTD_32','CeTD_33']
    for i in header2:
        for j in ('HB','VW','PO','PZ','CH','SS','SA'):
            head2.append(i+'_'+str(j))
    df12 = pd.read_csv("compout_2")
    df_2 = df12.iloc[:,:-1]
    ss = pd.DataFrame()
    for i in range(0,len(df_2),7):
        temp_ss = pd.DataFrame(pd.concat([df_2.loc[i],df_2.loc[i+1],df_2.loc[i+2],df_2.loc[i+3],df_2.loc[i+4],df_2.loc[i+5],df_2.loc[i+6]],axis=0)).transpose().reset_index(drop=True)
        ss = pd.concat([ss, temp_ss], ignore_index=True)
        #ss = ss.append(pd.DataFrame(pd.concat([df_2.loc[i],df_2.loc[i+1],df_2.loc[i+2],df_2.loc[i+3],df_2.loc[i+4],df_2.loc[i+5],df_2.loc[i+6]],axis=0)).transpose()).reset_index(drop=True)
    ss.columns = head2
        #ss.to_csv(filename+".ctd_trans", index=None, encoding='utf-8')
    head3 = []
    header3 = ['CeTD_0_p','CeTD_25_p','CeTD_50_p','CeTD_75_p','CeTD_100_p']
    header4 = ['HB','VW','PO','PZ','CH','SS','SA']
    for j in range(1,4):
        for k in header4:
            for i in header3:
                head3.append(i+'_'+k+str(j))
    df_3 = pd.read_csv("compout_3", sep=" ")
    rr = pd.DataFrame()
    for i in range(0,len(df_3),21):
        temp_rr = pd.DataFrame(pd.concat([df_3.loc[i],df_3.loc[i+1],df_3.loc[i+2],df_3.loc[i+3],df_3.loc[i+4],df_3.loc[i+5],df_3.loc[i+6],df_3.loc[i+7],df_3.loc[i+8],df_3.loc[i+9],df_3.loc[i+10],df_3.loc[i+11],df_3.loc[i+12],df_3.loc[i+13],df_3.loc[i+14],df_3.loc[i+15],df_3.loc[i+16],df_3.loc[i+17],df_3.loc[i+18],df_3.loc[i+19],df_3.loc[i+20]],axis=0)).transpose().reset_index(drop=True)
        rr = pd.concat([rr, temp_rr], ignore_index=True)
        #rr = rr.append(pd.DataFrame(pd.concat([df_3.loc[i],df_3.loc[i+1],df_3.loc[i+2],df_3.loc[i+3],df_3.loc[i+4],df_3.loc[i+5],df_3.loc[i+6],df_3.loc[i+7],df_3.loc[i+8],df_3.loc[i+9],df_3.loc[i+10],df_3.loc[i+11],df_3.loc[i+12],df_3.loc[i+13],df_3.loc[i+14],df_3.loc[i+15],df_3.loc[i+16],df_3.loc[i+17],df_3.loc[i+18],df_3.loc[i+19],df_3.loc[i+20]],axis=0)).transpose()).reset_index(drop=True)
    rr.columns = head3
    cotrdi= pd.concat([zz,ss,rr],axis=1)
    cotrdi.to_csv('sam_allcomp.ctd', index=None, encoding='utf-8')

# Function to read and implement the model
def model_run(file1,file2):
    a = []
    data_test = file1
    clf = pickle.load(open(file2,'rb'))
    y_p_score1=clf.predict_proba(data_test)
    y_p_s1=y_p_score1.tolist()
    a.extend(y_p_s1)
    df = pd.DataFrame(a)
    df1 = df.iloc[:,-1].round(2)
    df2 = pd.DataFrame(df1)
    df2.columns = ['ML_score']
    return df2

def MERCI_Processor_p(merci_file,merci_processed,df1):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(df1)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['0']))
            kk.append('Disease non-causing')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('Disease Causing')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('Disease non-causing')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_p(merci_processed,final_merci_p):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if df5['Hits'][i] > 0:
            kk.append(0.5)
        else:
            kk.append(0)
    df5["MERCI Score (+ve)"] = kk
    df5 = df5[['Subject','MERCI Score (+ve)']]
    df5.to_csv(final_merci_p, index=None)

def hybrid(mlres,df1, df2, merci_output_p, threshold,final_output):
    #df6_3 = pd.read_csv(mlres,header=None)
    df6_2 = pd.DataFrame(df2)
    df6_1 = pd.DataFrame(df1)
    df5 = pd.read_csv(merci_output_p, dtype={'Subject': object, 'MERCI Score': np.float64})
    #df4 = pd.read_csv(merci_output_n, dtype={'Subject': object, 'MERCI Score': np.float64})
    df6 = pd.concat([df6_1,df6_2, mlres],axis=1)
    df6.columns = ['Subject','Sequence','ML Score']
    df6['Subject'] = df6['Subject'].str.replace('>','')
    df7 = pd.merge(df6,df5, how='outer',on='Subject')
    #df8 = pd.merge(df7,df4, how='outer',on='Subject')
    df7.fillna(0, inplace=True)
    df7['Hybrid Score'] = df7[['ML Score', 'MERCI Score (+ve)']].sum(axis=1)
    df7 = df7.round(3)
    ee = []
    for i in range(0,len(df7)):
        if df7['Hybrid Score'][i] > float(threshold):
            ee.append('Disease Causing')
        else:
            ee.append('Disease non-causing')
    df7['Prediction'] = ee
    df7.to_csv(final_output, index=None)


# To Determine the label
def det_lab(file1,thr):
    df1 = file1
    df1['label'] = ['Disease Causing' if df1['ML_score'][i]>=float(thr) else 'Disease non-causing' for i in range(len(df1))]
    return df1

# Parameter initialization or assigning variable for command level arguments

input_file = args.input        # Input variable 
 
# Output file 
 
if args.output == None:
    result_filename= "outfile.csv" 
else:
    result_filename = args.output

# Job Type 
if args.job == None:
        Job = int(1)
else:
        Job = int(args.job)
# Window Length 
if args.winleng == None:
        Win_len = int(9)
else:
        Win_len = int(args.winleng)

# Threshold 
if args.threshold == None:
        Threshold = 0.72
else:
        Threshold= float(args.threshold)

# Display
if args.display == None:
        dplay = int(1)
else:
        dplay = int(args.display)

###########################################################################################
if Job==2:
    print("\n");
    print('##############################################################################')
    print('Summary of Parameters:')
    print('Input File: ',input_file,'; Threshold: ', Threshold,'; Job Type: ',Job)
    print('Output File: ',result_filename,'; Window Length: ',Win_len,'; Display: ',dplay)
    print('##############################################################################')

#####################################BLAST Path############################################
if os.path.exists('envfile'):
    with open('envfile', 'r') as file:
        data = file.readlines()
    output = []
    for line in data:
        if not "#" in line:
            output.append(line)
    if len(output)==4:
        paths = []
        for i in range (0,len(output)):
            paths.append(output[i].split(':')[1].replace('\n',''))
        merci = "MERCI_motif_locator.pl"
        motifs = "motifs"
       
    else:
        print("##############################################################################################################")
        print("Error: Please provide paths for PSI-BLAST, Swiss-Prot Database, and MERCI, and required files", file=sys.stderr)
        print("##############################################################################################################")
        sys.exit()
else:
    print("######################################################################################################")
    print("Error: Please provide the '{}', which comprises paths for PSI-BLAST".format('envfile'), file=sys.stderr)
    print("######################################################################################################")
    sys.exit()
###########################################################################################

print("\n");
print('##############################################################################')
print('Summary of Parameters:')
print('Input File: ',input_file,'; Threshold: ', Threshold,)
print('Output File: ',result_filename,'; Display: ',dplay)
print('# ############################################################################')


#======================= Prediction Module start from here =====================
if Job == 1:
    print('\n======= Thanks for using Predict module of RAIpred. Your results will be stored in file :',result_filename,' =====\n')
    df1, df2 = readseq(input_file)
    df3 = lenchk(df2)
    intermediate_file = "intermediate_input.fasta"
    df3.to_csv(intermediate_file, index=False, header=False)
    ctd(intermediate_file)
    X = pd.read_csv("sam_allcomp.ctd")
    X.to_csv('features.csv', index=False)
    #Run ML model and save result in output file provided by the user
    Threshold = 0.72
    mlres = model_run(X,'xgb_model_server')
    mlres = mlres.round(3)
    df45 = det_lab(mlres,Threshold)
    df44 = pd.concat([df1, df2, df45],axis=1)
    df44.columns = ['Seq_ID','Sequence','ML_Score','Prediction']
    df44.Seq_ID = df44.Seq_ID.str.replace('>','')
    if dplay == 1:
        df44 = df44.loc[df44.Prediction=="Disease Causing"]
    else:
        df44 = df44
    df44 = round(df44,3)
    df44.to_csv(result_filename, index=None)
    os.remove('intermediate_input.fasta')
    os.remove('compout_1')
    os.remove('compout_2')
    os.remove('sam_allcomp.ctd')
    os.remove('features.csv')
    os.remove('compout_3')
    print("\n=========Process Completed. Have an awesome day ahead.=============\n")    
#===================== Design Model Start from Here ======================
elif Job == 2:
    print('\n======= Thanks for using Design module of RAIpred. Your results will be stored in file :',result_filename,' =====\n')
    print('==== Designing Peptides: Processing sequences please wait ...')
    df1, df2 = readseq(input_file)
    df3 = lenchk(df2)
    df_1 = mutants(df3,df1)
    dfseq = df_1[['Seq']]
    intermediate_file = "intermediate_input.fasta"
    df3.to_csv(intermediate_file, index=False, header=False)
    ctd(intermediate_file)
    X = pd.read_csv("sam_allcomp.ctd")
    X.to_csv('features.csv', index=False)
    Threshold = 0.72
    mlres = model_run(X,'xgb_model_server')
    mlres = mlres.round(3)
    df45 = det_lab(mlres,Threshold)
    df45.columns = ['ML_Score','Prediction']
    df44 = pd.concat([df_1,df45],axis=1)
    df44['Mutant_ID'] = ['_'.join(df44['Mutant_ID'][i].split('_')[:-1]) for i in range(len(df44))]
    df44.drop(columns=['Seq_ID'],inplace=True)
    df44['Seq_ID'] = [i.replace('>','') for i in df_1['Seq_ID']]
    df44['Sequence'] = df_1.Seq
    df44 = df44[['Seq_ID','Mutant_ID','Sequence','ML_Score','Prediction']]
    df44.Seq_ID = df44.Seq_ID.str.replace('>','')
    if dplay == 1:
        df44 = df44.loc[df44.Prediction=="Disease Causing"]
    else:
        df44 = df44
    df44 = round(df44,3)
    df44.to_csv(result_filename, index=None)
    os.remove('intermediate_input.fasta')
    os.remove('compout_1')
    os.remove('compout_2')
    os.remove('sam_allcomp.ctd')
    os.remove('features.csv')
    os.remove('compout_3')
    print("\n=========Process Completed. Have an awesome day ahead.=============\n")
#=============== Scan Model start from here ==================
elif Job==3:
    print('\n======= Thanks for using Scan module of RAIpred. Your results will be stored in file :',result_filename,' =====\n')
    print('==== Scanning Peptides: Processing sequences please wait ...')
    df1, df2 = readseq(input_file)
    df3 = lenchk(df2)
    df_1 = seq_pattern(df3,df1,Win_len)
    dfseq = df_1[['Seq']]
    intermediate_file = "intermediate_input.fasta"
    df3.to_csv(intermediate_file, index=False, header=False)
    ctd(intermediate_file)
    X = pd.read_csv("sam_allcomp.ctd")
    X.to_csv('features.csv', index=False)
    Threshold = 0.72
    mlres = model_run(X,'xgb_model_server')
    mlres = mlres.round(3)
    df45 = det_lab(mlres,Threshold)
    df45.columns = ['ML_Score','Prediction']
    df44 = pd.concat([df_1,df45],axis=1)
    df44['Pattern_ID'] = ['_'.join(df44['Pattern_ID'][i].split('_')[:-1]) for i in range(len(df44))]
    df44.drop(columns=['Seq_ID'],inplace=True)
    df44['Seq_ID'] = [i.replace('>','') for i in df_1['Seq_ID']]
    df44['Sequence'] = df_1.Seq
    df44 = df44[['Seq_ID','Pattern_ID','Sequence','ML_Score','Prediction']]
    df44.Seq_ID = df44.Seq_ID.str.replace('>','')
    if dplay == 1:
        df44 = df44.loc[df44.Prediction=="Disease Causing"]
    else:
        df44 = df44
    df44 = round(df44,3)
    df44.to_csv(result_filename, index=None)
    os.remove('intermediate_input.fasta')
    os.remove('compout_1')
    os.remove('compout_2')
    os.remove('sam_allcomp.ctd')
    os.remove('features.csv')
    os.remove('compout_3')
    print("\n=========Process Completed. Have an awesome day ahead.=============\n")

#===================== Ensemble Model Start from Here ======================
elif Job == 4:
    print('\n======= Thanks for using Ensemble module of RAIPred. Your results will be stored in file :',result_filename,' =====\n')
    df1, df2 = readseq(input_file)
    df3 = lenchk(df2)
    intermediate_file = "intermediate_input.fasta"
    df3.to_csv(intermediate_file, index=False, header=False)
    ctd(intermediate_file)
    X = pd.read_csv("sam_allcomp.ctd")
    X.to_csv('features.csv', index=False)
    #Run ML model and save result in output file provided by the user
    Threshold = 0.72
    mlres = model_run(X,'xgb_model_server')
    mlres = mlres.round(3)
    subprocess.run([
    "perl", "MERCI_motif_locator.pl", 
    "-p", input_file, 
    "-i", motifs+"/pos_motif.txt", 
    "-c", "BETTS-RUSSELL", 
    "-o", "pos_hits.txt"
])
    MERCI_Processor_p( "pos_hits.txt",  'processed_motifs.csv',df1)
    Merci_after_processing_p( 'processed_motifs.csv',  'merci_hybrid_p.csv')
    hybrid( mlres,df1, df2,  'merci_hybrid_p.csv',Threshold,  'final_output')
    df44 = pd.read_csv( 'final_output')
    if dplay == 1:
        df44 = df44.loc[df44.Prediction=="Disease Causing"]
    else:
        df44 = df44
    df44 = round(df44,3)
    df44.to_csv(result_filename, index=None)

    os.remove('intermediate_input.fasta')
    os.remove('compout_1')
    os.remove('compout_2')
    os.remove('sam_allcomp.ctd')
    os.remove('features.csv')
    os.remove('pos_hits.txt')
    os.remove('processed_motifs.csv')
    os.remove('merci_hybrid_p.csv')
    os.remove('final_output')
    os.remove('compout_3')
    print("\n=========Process Completed. Have an awesome day ahead.=============\n")
print('\n======= Thanks for using RAIpred. Your results are stored in file :',result_filename,' =====\n\n')





