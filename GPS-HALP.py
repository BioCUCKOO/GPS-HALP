# -*- coding: utf-8 -*-
# @Time    : 2023/12/19  
# @File    : GPS-HALP.py


import sys, os, re, joblib
import pandas as pd
import numpy as np
import math
from collections import Counter, OrderedDict
from keras.models import load_model,Model,model_from_json
import keras.backend as KTF
import gc
def fasta2seq_aa(input):
    #将输入的fasta文件输出为psp的csv文件
    f_fas = open(input, 'r')
    id_seq = {}
    global linename
    for line in f_fas:
        if line.startswith('>'):
           linename = line.replace('>', '').strip()
           id_seq[linename] = ''
        else:
           id_seq[linename] += line.strip()
    human_p = open(r"/home/biocuckoo/gps-halp/code_python/database/seq2hpa.tsv",'r')
    line = human_p.readline()
    huamn_p_dic = {}
    while(line):
        items = line.strip().split("\t")
        huamn_p_dic[items[1]] = items[0]
        line = human_p.readline()
    huamn_p_id = {}
    for linename in id_seq.keys():
        if id_seq[linename] in huamn_p_dic.keys():
            huamn_p_id[linename] = huamn_p_dic[id_seq[linename]]
        else:
            huamn_p_id[linename] = "NA"

           
    H_ids = []
    K_ids = []
    R_ids = []
    H_psp21 = []
    K_psp21 = []
    R_psp21 = []
    H_locs = []
    K_locs = []
    R_locs = []
    for k,v in id_seq.items():
        H_psps = []
        K_psps = []
        R_psps = []
        proposal_psp = '**********'  # 30个
        for i in range(len(v)):
            if  v[i] == "H":

                if i < 10:
                    # 按照位置补够氨基酸前面的空位
                    psp = proposal_psp[:(10 - i)]
                    tem = v[:i + 11]
                    psp = psp + tem
                    if len(psp) < 21:
                        tem2 = proposal_psp[:(21 - len(psp))]
                        psp = psp + tem2
                # 末尾的氨基酸
                elif i >= (len(v) - 10):
                    psp = v[(i - 10):(len(v))]
                    tem = proposal_psp[:(10 - (len(v) - i - 1))]
                    psp = psp + tem
                else:
                    psp = v[i - 10:i + 11]
                H_psps.append(psp)
                H_locs.append(i)

            elif v[i]=="K":
                
                if i < 10:
                    # 按照位置补够氨基酸前面的空位
                    psp = proposal_psp[:(10 - i)]
                    tem = v[:i + 11]
                    psp = psp + tem
                    if len(psp) < 21:
                        tem2 = proposal_psp[:(21 - len(psp))]
                        psp = psp + tem2
                # 末尾的氨基酸
                elif i >= (len(v) - 10):
                    psp = v[(i - 10):(len(v))]
                    tem = proposal_psp[:(10 - (len(v) - i - 1))]
                    psp = psp + tem
                else:
                    psp = v[i - 10:i + 11]
                K_psps.append(psp)
                K_locs.append(i)

            elif v[i]=="R":
                if i < 10:
                    # 按照位置补够氨基酸前面的空位
                    psp = proposal_psp[:(10 - i)]
                    tem = v[:i + 11]
                    psp = psp + tem
                    if len(psp) < 21:
                        tem2 = proposal_psp[:(21 - len(psp))]
                        psp = psp + tem2
                # 末尾的氨基酸
                elif i >= (len(v) - 10):
                    psp = v[(i - 10):(len(v))]
                    tem = proposal_psp[:(10 - (len(v) - i - 1))]
                    psp = psp + tem
                else:
                    psp = v[i - 10:i + 11]
                R_psps.append(psp)
                R_locs.append(i)

        H_ids.extend([k]*len(H_psps))
        K_ids.extend([k]*len(K_psps))
        R_ids.extend([k]*len(R_psps))
        H_psp21.extend(H_psps)
        K_psp21.extend(K_psps)
        R_psp21.extend(R_psps)
    return huamn_p_id, H_ids,K_ids,R_ids,H_psp21,K_psp21,R_psp21,H_locs,K_locs,R_locs
#f1 GPS
def getArray(l_AAs):
    score = []
    for i in range(len(l_AAs)):
        score.append(0.0)
    scoreary = np.array(score)
    return scoreary

def get_all_pos(input_seq):
    file_pos = open(r"/home/biocuckoo/gps-halp/code_python/GPS_pos.txt",'r')
    lines = file_pos.readlines()
    positive_seqs = []
    for line in lines:
        if ">" not in line:
            seq = line.strip()
            seq = seq.replace('U',"*")
            positive_seqs.append(seq)

    file_Blosum62 = open(r"/home/biocuckoo/gps-halp/code_python/BLOSUM62_NCBI.txt",'r')
    lines = file_Blosum62.readlines()
    title_items = lines[1].strip().split(",")#所有氨基酸
    BLOSUM62_dic = {}
    aa = []
    for line in lines[2:]:
        items = line.strip().split(",")
        aa.append(items[0])#所有氨基酸
        for i in range(1,len(items)):
            key = items[0]+title_items[i]#遍历len(items)行之前的所有氨基酸对
            if key not in BLOSUM62_dic.keys():
                if "*" not in key:
                    BLOSUM62_dic[key] = int(items[i])
                else:
                    BLOSUM62_dic[key] = 0#*和氨基酸的对重新赋值为0
                #BLOSUM62_dic[key] = int(items[i])

    train_x = positive_seqs + input_seq 
    train_y = [1]*len(positive_seqs) + [0]*len(input_seq)#标签
    for i in range(len(input_seq)):
        if input_seq[i] in positive_seqs:
            train_y[len(positive_seqs)+i] = 1
    
    positive_seq_num = len(positive_seqs)
    positive_position_aa = {}#aa composition of each position in positive data
    for i in range(21):
        positive_position_aa[i] = {}
        for item in aa:#单个氨基酸
            positive_position_aa[i][item] = 0
        aa_seq = [x[i] for x in positive_seqs]#遍历所有阳性肽段的21个位置
        for item in aa_seq:
            positive_position_aa[i][item] += 1#21个位置上，如果有某个氨基酸，则个数加一；第一维是21个位置，第二维是氨基酸
                
    return(train_x,train_y,positive_position_aa,BLOSUM62_dic,aa,positive_seq_num)


def encode_GPS(input_seq, train_x, train_y, positive_position_aa, BLOSUM62_dic, AA, positive_seq_num, weight, matrix_weight):

    positive_matrix_score_dic = {}
    for i in range(21):
        positive_matrix_score_dic[i] = {}
        for item in AA:
            positive_matrix_score_dic[i][item] = {}
        for item in AA:
            for item2 in AA:
                aa_index = AA.index(item) * len(AA) + AA.index(item2)
                temp_score = positive_position_aa[i][item2] * \
                    BLOSUM62_dic[item+item2]*weight[i]*matrix_weight[aa_index]
                positive_matrix_score_dic[i][item][item2] = temp_score

    train_feature = []
    max_percentage = 0
    for i in range(7351,len(train_x)):
        minus_tag = train_y[i]
        temp_score = []
        for pos in range(21):
            this_aa = train_x[i][pos]
            pre_score = [0]*(25*25)
            if minus_tag == 1:
                pos_score_list = positive_matrix_score_dic[pos][this_aa]
                for a in AA:
                    aa_index = AA.index(this_aa) * len(AA) + AA.index(a)
                    sustitution_score = pos_score_list[a] - \
                        BLOSUM62_dic[this_aa+a]*weight[pos] * \
                        matrix_weight[aa_index]
                    pre_score[aa_index] = sustitution_score
            else:
                for a in AA:
                    aa_index = AA.index(this_aa) * len(AA) + AA.index(a)
                    pre_score[aa_index] = positive_matrix_score_dic[pos][this_aa][a]
            temp_score.append(pre_score)
        final_score = []
        for pos in range(len(temp_score[0])):
            temp = sum([x[pos] for x in temp_score])
            final_score.append(temp)
        final_score_re = [x/(positive_seq_num-minus_tag) for x in final_score]
        train_feature.append(final_score_re)
    train_feature_arr = np.array(train_feature)
    
    return(train_feature_arr)

def get_GPS(input_seq):
    train_x, train_y, positive_position_aa, BLOSUM62_dic, AA, positive_seq_num = get_all_pos(input_seq)
    
    with open(r"/home/biocuckoo/gps-halp/code_python/GPS_MatrixRenew.txt") as f:
        line = f.readline()
        line = f.readline()
        weight = [float(w) for w in line.strip('\n').split('\t')]
        line = f.readline()
        line = f.readline()
        matrix_weight = [float(w) for w in line.strip('\n').split('\t')]

    train_feature_arr = encode_GPS(input_seq, train_x, train_y, positive_position_aa, BLOSUM62_dic, AA, positive_seq_num, weight, matrix_weight)
    return(train_feature_arr)

# f2 AAC
def get_AAC(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []

    for i in seq:
        sequence = re.sub('\*', '', i)
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    np_results = np.array(encodings)
    return(np_results)
# f3 SOCNumber
def get_SOCNumber(seq):
	
    dataFile = r"/home/biocuckoo/gps-halp/code_python/Schneider-Wrede.txt"
    dataFile1 = r"/home/biocuckoo/gps-halp/code_python/Grantham.txt"
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA1 = 'ARNDCQEGHILKMFPSTWYV'

    DictAA = {}
    for i in range(len(AA)):
        DictAA[AA[i]] = i

    DictAA1 = {}
    for i in range(len(AA1)):
        DictAA1[AA1[i]] = i

    with open(dataFile) as f:
        records = f.readlines()[1:]
    AADistance = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance.append(array)
    AADistance = np.array(
        [float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))

    with open(dataFile1) as f:
        records = f.readlines()[1:]
    AADistance1 = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AADistance1.append(array)
    AADistance1 = np.array(
        [float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
        (20, 20))

    encodings = []

    for i in seq:
        sequence = re.sub('\*', '', i)
        code = []
        for n in range(1, 4):
            code.append(sum(
                [AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]) / (
                        len(sequence) - n))

        for n in range(1, 4):
            code.append(sum([AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in
                                range(len(sequence) - n)]) / (len(sequence) - n))
        encodings.append(code)
    np_results = np.array(encodings)
    return(np_results)

#f4 PAAC
def PAAC_Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def get_PAAC(seq, w=0.05):

    dataFile = r"/home/biocuckoo/gps-halp/code_python/PAAC.txt"
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
        AAProperty1.append([(j-meanI)/fenmu for j in i])

    encodings = []

    for i in seq:
        sequence = re.sub('\*', '', i)
        code = []
        theta = []
        for n in range(1, 3):
            theta.append(
                sum([PAAC_Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        w= 0.05
        code = code + [myDict[aa] / (1 + w * sum(map(float,theta))) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    np_results = np.array(encodings)
    return(np_results)

#f5 AC
def get_AC(seq, props=['ANDN920101', 'ARGP820101', 'ARGP820102', 'ARGP820103','BEGF750101', 'BEGF750102', 'BEGF750103', 'BHAR880101']):

        AA = 'ARNDCQEGHILKMFPSTWYV'
        fileAAidx = r"/home/biocuckoo/gps-halp/code_python/AAidx.txt"
        with open(fileAAidx) as f:
            records = f.readlines()[1:]
        myDict = {}
        for i in records:
            array = i.rstrip().split('\t')
            myDict[array[0]] = array[1:]

        AAidx = []
        AAidxName = []
        for i in props:
            if i in myDict:
                AAidx.append(myDict[i])
                AAidxName.append(i)
            else:
                print('"' + i + '" properties not exist.')
                return None

        AAidx1 = np.array([float(j) for i in AAidx for j in i])
        AAidx = AAidx1.reshape((len(AAidx), 20))

        propMean = np.mean(AAidx, axis=1)
        propStd = np.std(AAidx, axis=1)

        for i in range(len(AAidx)):
            for j in range(len(AAidx[i])):
                AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]

        index = {}
        for i in range(len(AA)):
            index[AA[i]] = i

        encodings = []
        for i in seq:  
            sequence = re.sub('\*','',i)
            code = []
            N = len(sequence)
            for prop in range(len(props)):
                xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
                for n in range(1, 3):
                    if len(sequence) > 2:
                        rn = 1/((N-n)) * sum(([(AAidx[prop][index.get(sequence[j + n],0)] - np.mat(xmean))])*(AAidx[prop][index.get(sequence[j], 0)] - np.mat(xmean)) for j in range(len(sequence)-n))
                    else:
                        rn = 'NA'
                    code.append(rn)
            encodings.append(code)
        new_list = []

        # 遍历原始列表
        for sublist in encodings:
            new_sublist = [matrix.item() for matrix in sublist]     
            new_list.append(new_sublist)
        np_results = np.array(new_list)
        return(np_results)

#f6 binary3bit1
def seq2bit3_type1(psp):
    dic = {'A': [0, 1, 0], 'C': [0, 0, 1], 'D': [1, 0, 0], 'E': [1, 0, 0], 'F': [0, 0, 1],
           'G': [0, 1, 0],
           'H': [0, 1, 0], 'I': [0, 0, 1], 'K': [1, 0, 0], 'L': [0, 0, 1], 'M': [0, 0, 1],
           'N': [1, 0, 0],
           'P': [0, 1, 0], 'Q': [1, 0, 0], 'R': [1, 0, 0], 'S': [0, 1, 0], 'T': [0, 1, 0],
           'V': [0, 0, 1],
           'W': [0, 0, 1], 'Y': [0, 1, 0], '*': [0, 0, 0]}
    psp = psp.strip()
    bn3_1 = []
    for aa in psp:
        if aa not in dic.keys():
            aa = '*'
        tl = dic[aa]
        bn3_1 = bn3_1 + tl
    return bn3_1

def get_binary3bit1(seq):
    results = list(map(seq2bit3_type1, seq))
    np_results = np.array(results)
    return(np_results)

#f7 binary3bit5
def seq2bit3_type5(psp):
    dic = {'A': [0, 1, 0], 'C': [0, 1, 0], 'D': [0, 0, 1], 'E': [0, 0, 1], 'F': [0, 1, 0],
           'G': [0, 1, 0],
           'H': [0, 1, 0], 'I': [0, 1, 0], 'K': [1, 0, 0], 'L': [0, 1, 0], 'M': [0, 1, 0],
           'N': [0, 1, 0],
           'P': [0, 1, 0], 'Q': [0, 1, 0], 'R': [1, 0, 0], 'S': [0, 1, 0], 'T': [0, 1, 0],
           'V': [0, 1, 0],
           'W': [0, 1, 0], 'Y': [0, 1, 0], '*': [0, 0, 0]}
    psp = psp.strip()
    bn3_5 = []
    for aa in psp:
        if aa not in dic.keys():
            aa = '*'
        tl = dic[aa]
        bn3_5 = bn3_5 + tl
    return bn3_5

def get_binary3bit5(seq):
    results = list(map(seq2bit3_type5, seq))
    np_results = np.array(results)
    return(np_results)

#f8 AESNN3
def seq2AESNN3(psp):
    dic = {'A': [-0.99, -0.61, 0.00],
           'R': [0.28, -0.99, -0.22],
           'N': [0.77, -0.24, 0.59],
           'D': [0.74, -0.72, -0.35],
           'C': [0.34, 0.88, 0.35],
           'Q': [0.12, -0.99, -0.99],
           'E': [0.59, -0.55, -0.99],
           'G': [-0.79, -0.99, 0.10],
           'H': [0.08, -0.71, 0.68],
           'I': [-0.77, 0.67, -0.37],
           'L': [-0.92, 0.31, -0.99],
           'K': [-0.63, 0.25, 0.50],
           'M': [-0.80, 0.44, -0.71],
           'F': [0.87, 0.65, -0.53],
           'P': [-0.99, -0.99, -0.99],
           'S': [0.99, 0.40, 0.37],
           'T': [0.42, 0.21, 0.97],
           'W': [-0.13, 0.77, -0.90],
           'Y': [0.59, 0.33, -0.99],
           'V': [-0.99, 0.27, -0.52],
           '*': [0, 0, 0]}
    psp = psp.strip()
    opf7 = []
    for aa in psp:
        if aa not in dic.keys():
            aa = '*'
        tl = dic[aa]
        opf7 = opf7 + tl
    return opf7


def get_AESNN3(seq):
    results = list(map(seq2AESNN3, seq))
    np_results = np.array(results)
    return(np_results)

#f9 PseKRAAC
def gapModel(seq, myDict, gDict, gNames, glValue):
	encodings = []
	for i in seq:
		sequence = re.sub('\*', '', i)
		code = []
		numDict = {}
		for j in range(0, len(sequence), glValue + 1):
			if j+1 < len(sequence):
				numDict[gDict[myDict[sequence[j]]]+'_'+gDict[myDict[sequence[j+1]]]] = numDict.get(gDict[myDict[sequence[j]]]+'_'+gDict[myDict[sequence[j+1]]], 0) + 1

		for g in [g1+'_'+g2 for g1 in gNames for g2 in gNames]:
			code.append(numDict.get(g, 0))
		encodings.append(code)
	return encodings


def get_PseKRAAC(seq):
    AAGroup = {
	3:['FWYCILMVAGSTPHNQ', 'DE', 'KR'],
	4:['FWY', 'CILMV', 'AGSTP', 'EQNDHKR'],
	8:['FWY', 'CILMV', 'GA', 'ST', 'P', 'EQND', 'H', 'KR'],
	10:['G', 'FYW', 'A', 'ILMV', 'RK', 'P', 'EQND', 'H', 'ST', 'C'],
	15:['G', 'FY', 'W', 'A', 'ILMV', 'E', 'Q', 'RK', 'P', 'N', 'D', 'H', 'S', 'T', 'C'],
	20:['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C'],
        }
    # index each amino acids to their group
    myDict = {}
    for i in range(len(AAGroup[3])):
        for aa in AAGroup[3][i]:
            myDict[aa] = i

    gDict = {}
    gNames = []
    for i in range(len(AAGroup[3])):
        gDict[i] = 'T5.G.'+str(i+1)
        gNames.append('T5.G.'+str(i+1))

    encodings = []
    encodings = gapModel(seq, myDict, gDict, gNames, 2)

    np_results = np.array(encodings)
    return(np_results)

#f10 GDPC
def get_GDPC(seq):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    baseNum = len(groupKey)
    dipeptide = [g1+'.'+g2 for g1 in groupKey for g2 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []
    for i in seq:
        sequence = re.sub('\*', '', i)

        code = []
        myDict = {}
        for t in dipeptide:
            myDict[t] = 0

        sum = 0
        for j in range(len(sequence) - 2 + 1):
            myDict[index[sequence[j]]+'.'+index[sequence[j+1]]] = myDict[index[sequence[j]]+'.'+index[sequence[j+1]]] + 1
            sum = sum +1

        if sum == 0:
            for t in dipeptide:
                code.append(0)
        else:
            for t in dipeptide:
                code.append(myDict[t]/sum)
        encodings.append(code)
    np_results = np.array(encodings)
    return(np_results)

def seq2fts(seq):
    f1 = get_GPS(seq)
    f2 = get_AAC(seq)
    f3 = get_SOCNumber(seq)
    f4 = get_PAAC(seq)
    f5 = get_AC(seq)
    f6 = get_binary3bit1(seq)
    f7 = get_binary3bit5(seq)
    f8 = get_AESNN3(seq)
    f9 = get_PseKRAAC(seq)
    f10 = get_GDPC(seq)
    return [f2,f8,f5,f6,f7,f10,f1,f4,f9,f3]


def test_predict(feature_path, model_path):
    feature_list = ['AAC','AESNN3', 'AC', 'binary3bit1', 'binary3bit5', 'GDPC', 'GPS', 'PAAC','PseKRAAC', 'SOCNumber']  #
    pred = []
    for i in range(10):
        fn = feature_list[i]
        KTF.clear_session()
        json_file = open(model_path + 'Ensemble/' +fn + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        DNN_model = model_from_json(loaded_model_json)
        DNN_model.load_weights(model_path + 'Ensemble/' +fn + '.h5')
        DNN_feature = DNN_model.predict(feature_path[i])
        
        KTF.clear_session()
        json_file = open(model_path + 'AutoEncoder/' +fn + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        AE_model = model_from_json(loaded_model_json)
        AE_model.load_weights(model_path + 'AutoEncoder/' +fn + '.h5')
        layer_name = "encoder"
        intermediate_layer_model = Model(inputs=AE_model.input,outputs=AE_model.get_layer(layer_name).output)
        AE_feature = intermediate_layer_model.predict(feature_path[i])
        pred.append(DNN_feature)
        pred.append(AE_feature)

        del AE_model
        del intermediate_layer_model
        del DNN_model
        gc.collect()
    test_x_Mix = pred[0]
    for j in range(1,len(pred)):
        test_x_Mix = np.hstack((test_x_Mix,pred[j]))
        
    json_file = open(model_path + 'MixModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    comb_model = model_from_json(loaded_model_json)
    comb_model.load_weights(model_path + 'MixModel.h5')
    pred_comb = comb_model.predict(test_x_Mix)
       
    return pred_comb

def out_format(pred_score, pred_type, threshold, psps, ids, kn, locs, huamn_p_id):
    
    file = open(r"/home/biocuckoo/gps-halp/code_python/database/verified_sites.txt",'r')
    lines = file.readlines()
    database_sites = {}
    for line in lines:
        items = line.strip("\n").split("\t")
        database_sites[items[0]+"_"+items[1]] = items[2]
    database_sites['NA'] = '-'
    file.close()

    file = open(r"/home/biocuckoo/gps-halp/Basic/Protein-View-0.txt",'r')
    lines = file.readlines()
    database_id = {}
    for line in lines:
        items = line.strip("\n").split("\t")
        database_id[items[1]] = items[0]
    database_id['NA'] = '-'
    file.close()
    
    file = open(r"/home/biocuckoo/gps-halp/code_python/database/HPA_tissue_exp_site.txt",'r')
    lines = file.readlines()
    database = {}
    for line in lines:
        items = line.strip("\n").split("\t")
        wrt_str = []
        for i in range(2,len(items),2):
            wrt_str.append(items[i].capitalize()+" ({0})".format(items[i+1]))
        database[items[0]] = [";".join(wrt_str),items[1]]
    database['NA'] = ['-','-']
    file.close()

    file = open(r"/home/biocuckoo/gps-halp/code_python/database/imm_exp_uni.txt",'r')
    lines = file.readlines()
    database_immune = {}
    for line in lines[1:]:
        items = line.strip("\n").split("\t")
        wrt_str = []
        for i in range(2,len(items)-1):
            wrt_str.append(items[i].split(":")[1].replace("_"," ")+" ({0})".format(items[i].split(":")[0]))
        database_immune[items[0]] = [";".join(wrt_str),items[1]]
    database_immune['NA'] = ['-','-']
    file.close()

    file = open(r"/home/biocuckoo/gps-halp/code_python/database/cancer_merge.txt",'r')
    lines = file.readlines()
    database_mut = {}
    for line in lines[1:]:
        items = line.strip("\n").split("\t")
        wrt_top = []
        wrt_str = []
        for i in range(1,len(items)):
            sub_items = items[i].split(";")
            new_sub_items = []
            for item in sub_items:
                if item != '':
                    item_items = item.split("|")
                    samples = item_items[3].split("+")
                    if len(samples) > 3:
                        item_items[3] = "+".join(samples[0:3])
                    if i == 1:
                        new_sub_items.append("COSMIC@{0}@{1}@{2}@{3}".format(item_items[2],item_items[0],item_items[1],item_items[3].replace("+","|")))
                    elif i == 2:
                        new_sub_items.append("ICGC@{0}@{1}@{2}@{3}".format(item_items[2],item_items[0],item_items[1],item_items[3].replace("+","|")))
                    elif i == 3:
                        new_sub_items.append("TCGA@{0}@{1}@{2}@{3}".format(item_items[2],item_items[0],item_items[1],item_items[3].replace("+","|")))
                wrt_str+=new_sub_items
        database_mut[items[0]] = ";".join(wrt_str)
    file.close()
    
    out1 = open(outf, 'a+')
    size = os.path.getsize(outf)
    if size == 0:
        out1.write("ID\tPosition\tCode\tPeptide\tScore\tCutoff\tSource\tHPA Tissue\tHPA Link\tImmune Cell\tImmune Cell Link\tMutation\tUniprot\n")
    id2 = ids[0].strip()
    print("pred_type",len(pred_type),"ids",len(ids))
    for i in range(len(pred_type)):
        if pred_type[i]:
            uni_id = huamn_p_id[ids[i]]
            if uni_id+"_"+str(locs[i] + 1) in database_sites.keys():
                source_str = database_sites[uni_id+"_"+str(locs[i] + 1)]
            else:
                source_str = "-"

            if uni_id in database_id.keys():
                id_str = database_id[uni_id]
            else:
                id_str = "-"

            if uni_id+"_"+str(locs[i] + 1) in database_mut.keys():
                tem1 = ids[i].strip() + '\t' + str(locs[i] + 1) + '\t' + psps[i][10] + '\t' + psps[i][3:18] + '\t' + str(round(pred_score[i][0],4)) + '\t' + str(threshold) + '\t' + source_str + "\t" + database[uni_id][0] + '\t' + database[uni_id][1] + "\t" + database_immune[uni_id][0] + '\t' + database_immune[uni_id][1] + "\t" + database_mut[uni_id+"_"+str(locs[i] + 1)] + "\t" + id_str + '\n'
            else:
                tem1 = ids[i].strip() + '\t' + str(locs[i] + 1) + '\t' + psps[i][10] + '\t' + psps[i][3:18] + '\t' + str(round(pred_score[i][0],4)) + '\t' + str(threshold) + '\t' + source_str + "\t" + database[uni_id][0] + '\t' + database[uni_id][1] + "\t" + database_immune[uni_id][0] + '\t' + database_immune[uni_id][1] + "\t"+ "-" + "\t" + id_str + '\n'
            out1.write(tem1)
    out1.close()


if __name__ == '__main__':

    threshold= sys.argv[1]
    treeNode = sys.argv[2]
    upfile =sys.argv[3]     
    outpath =sys.argv[4]

    outf = outpath
    outpath = outpath.rsplit('.gps', 1)[0] + '_'

    GS = treeNode.split(",")
    GS = [x for x in GS]
    G0 = GS[0]

    huamn_p_id, H_ids,K_ids,R_ids,H_psp21,K_psp21,R_psp21,H_locs,K_locs,R_locs = fasta2seq_aa(upfile)

    data_dic = {"His":[H_ids,H_psp21,H_locs],"Arg":[R_ids,R_psp21,R_locs],"Lys":[K_ids,K_psp21,K_locs]}
    for nphosaa in GS:
        print(nphosaa)
        featuresall = seq2fts(data_dic[nphosaa[:-1]][1])
        knpath = "/home/biocuckoo/gps-halp/webcomp/models/AA/" + nphosaa
        pred_comb = test_predict(featuresall, knpath)
        zhibiao = pd.read_csv( knpath+ "zhibiao.txt", sep="\t", header=None)
        if threshold == "a":
            thd = "Null"
            pred_type = [1] * len(pred_comb)
            out_format(pred_comb.tolist(), pred_type, thd, data_dic[nphosaa[:-1]][1], data_dic[nphosaa[:-1]][0], nphosaa, data_dic[nphosaa[:-1]][2], huamn_p_id)
        else:
            thd = 0
            if threshold == "h":
                thd = round(float(zhibiao.iloc[0, 1]), 4)
            elif threshold == "m":
                thd = round(float(zhibiao.iloc[1, 1]), 4)
            elif threshold == "l":
                thd = round(float(zhibiao.iloc[2, 1]), 4)
            pred_type = (pred_comb >= thd).astype(bool)
            out_format(pred_comb.tolist(), pred_type, thd, data_dic[nphosaa[:-1]][1], data_dic[nphosaa[:-1]][0], nphosaa, data_dic[nphosaa[:-1]][2], huamn_p_id)
    sys.exit()

