# -*- coding: utf-8 -*-

import sys, os, re, joblib
import pandas as pd
import numpy as np
import math
from collections import Counter, OrderedDict
from keras.models import load_model,Model,model_from_json
import keras.backend as KTF
import gc
input_seq = ["FDAKAKEFIAKLQANPAKIAS"]
def fasta2seq_aa(OO0OOO0OO0OOOOOOO):
    #将输入的fasta文件输出为psp的csv文件
    OO0OOO0OO0OOOOOOO = open(OO0OOO0OO0OOOOOOO, 'r')
    OO0OOO0OO0OOOOOOO = {}
    global linename
    for OO0OOO0OO0OOOOOOO in OO0OOO0OO0OOOOOOO:
        if OO0OOO0OO0OOOOOOO.startswith('>'):
           linename = OO0OOO0OO0OOOOOOO.replace('>', '').strip()
           OO0OOO0OO0OOOOOOO[linename] = ''
        else:
           OO0OOO0OO0OOOOOOO[linename] += OO0OOO0OO0OOOOOOO.strip()
    #开始匹配人的序列
    OO0OOO0OO0OOOOOO0 = open(r"/home/biocuckoo/public_html/gps-halp/code_python/database/seq2hpa.tsv",'r')
    OO0OOO0OO0OOOOO00= OO0OOO0OO0OOOOOO0.readline()
    OO0OOO0OO0OOOO000 = {}
    while(OO0OOO0OO0OOOOO00):
        items = OO0OOO0OO0OOOOO00.strip().split("\t")
        OO0OOO0OO0OOOO000[items[1]] = items[0]
        OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOOO0.readline()
    OO0OOO0OO0OOOOOOO = {}
    for OO0OOO0OO0OOO0000 in OO0OOO0OO0OOOOOOO.keys():
        if OO0OOO0OO0OOOOOOO[OO0OOO0OO0OOO0000] in OO0OOO0OO0OOOO000.keys():
            OO0OOO0OO0OOOOOOO[OO0OOO0OO0OOO0000] = OO0OOO0OO0OOOO000[OO0OOO0OO0OOOOOOO[linename]]
        else:
            OO0OOO0OO0OOOOOOO[OO0OOO0OO0OOO0000] = "NA"

           
    OO0OOO0OO0OOOOOOO = []
    OO0OOO0OO0OOOOOOO = []
    OO0OOO0OO0OOOOOOO = []
    OO0OOO0OO0OOOOOOO = []
    OO0OOO0OO0OOOOOOO = []
    OO0OOO0OO0OOOOOOO = []
    OO0OOO0OO0OOOOOOO = []
    OO0OOO0OO0OOOOOOO = []
    OO0OOO0OO0OOOOOOO = []
    for OO0OOO0O00OOOOO00,OO0O0O0O00OOOOO00 in OO0OOO0OO0OOOOOOO.items():
        # print(OO0OOO0O00OOOOO00,OO0O0O0O00OOOOO00)
        OO0OOO0OO0OOOOOOO = []
        OO0OOO0OO0OOOOOOO = []
        OO0OOO0OO0OOOOOOO = []
        OO0OOO0OO0OO00000 = '**********'  # 30个
        for i in range(len(OO0O0O0O00OOOOO00)):
            if  OO0O0O0O00OOOOO00[i] == "H":

                if i < 10:
                    # 按照位置补够氨基酸前面的空位
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OO00000[:(10 - i)]
                    tem = OO0O0O0O00OOOOO00[:i + 11]
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem
                    if len(OO0OOO0OO0OOOOO00) < 21:
                        tem2 = OO0OOO0OO0OO00000[:(21 - len(OO0OOO0OO0OOOOO00))]
                        OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem2
                # 末尾的氨基酸
                elif i >= (len(OO0O0O0O00OOOOO00) - 10):
                    OO0OOO0OO0OOOOO00 = OO0O0O0O00OOOOO00[(i - 10):(len(OO0O0O0O00OOOOO00))]
                    tem = OO0OOO0OO0OO00000[:(10 - (len(OO0O0O0O00OOOOO00) - i - 1))]
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem
                else:
                    OO0OOO0OO0OOOOO00 = OO0O0O0O00OOOOO00[i - 10:i + 11]
                OO0OOO0OO0OOOOOOO.append(OO0OOO0OO0OOOOO00)
                OO0OOO0OO0OOOOOOO.append(i)

            elif OO0O0O0O00OOOOO00[i]=="K":
                
                if i < 10:
                    # 按照位置补够氨基酸前面的空位
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OO00000[:(10 - i)]
                    tem = OO0O0O0O00OOOOO00[:i + 11]
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem
                    if len(OO0OOO0OO0OOOOO00) < 21:
                        tem2 = OO0OOO0OO0OO00000[:(21 - len(OO0OOO0OO0OOOOO00))]
                        OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem2
                # 末尾的氨基酸
                elif i >= (len(OO0O0O0O00OOOOO00) - 10):
                    OO0OOO0OO0OOOOO00 = OO0O0O0O00OOOOO00[(i - 10):(len(OO0O0O0O00OOOOO00))]
                    tem = OO0OOO0OO0OO00000[:(10 - (len(OO0O0O0O00OOOOO00) - i - 1))]
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem
                else:
                    OO0OOO0OO0OOOOO00 = OO0O0O0O00OOOOO00[i - 10:i + 11]
                OO0OOO0OO0OOOOOOO.append(OO0OOO0OO0OOOOO00)
                OO0OOO0OO0OOOOOOO.append(i)

            elif OO0O0O0O00OOOOO00[i]=="R":
                if i < 10:
                    # 按照位置补够氨基酸前面的空位
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OO00000[:(10 - i)]
                    tem = OO0O0O0O00OOOOO00[:i + 11]
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem
                    if len(OO0OOO0OO0OOOOO00) < 21:
                        tem2 = OO0OOO0OO0OO00000[:(21 - len(OO0OOO0OO0OOOOO00))]
                        OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem2
                # 末尾的氨基酸
                elif i >= (len(OO0O0O0O00OOOOO00) - 10):
                    OO0OOO0OO0OOOOO00 = OO0O0O0O00OOOOO00[(i - 10):(len(OO0O0O0O00OOOOO00))]
                    tem = OO0OOO0OO0OO00000[:(10 - (len(OO0O0O0O00OOOOO00) - i - 1))]
                    OO0OOO0OO0OOOOO00 = OO0OOO0OO0OOOOO00 + tem
                else:
                    OO0OOO0OO0OOOOO00 = OO0O0O0O00OOOOO00[i - 10:i + 11]
                OO0OOO0OO0OOOOOOO.append(OO0OOO0OO0OOOOO00)
                OO0OOO0OO0OOOOOOO.append(i)

        OO0OOO0OO0OOOOOOO.extend([OO0OOO0O00OOOOO00]*len(OO0OOO0OO0OOOOOOO))
        OO0OOO0OO0OOOOOOO.extend([OO0OOO0O00OOOOO00]*len(OO0OOO0OO0OOOOOOO))
        OO0OOO0OO0OOOOOOO.extend([OO0OOO0O00OOOOO00]*len(OO0OOO0OO0OOOOOOO))
        OO0OOO0OO0OOOOOOO.extend(OO0OOO0OO0OOOOOOO)
        OO0OOO0OO0OOOOOOO.extend(OO0OOO0OO0OOOOOOO)
        OO0OOO0OO0OOOOOOO.extend(OO0OOO0OO0OOOOOOO)
    return OO0OOO0OO0OOOOOOO, OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO
#f1 GPS
def getArray(O0000O0OO0O0O00O0):
    O0OO0O0O0OO0000OO = []
    for O0O0O000O00OOOOOO in range(len(O0000O0OO0O0O00O0)):
        O0OO0O0O0OO0000OO.append(0.0)
    OOOO0OOO000OOOO0O = np.array(O0OO0O0O0OO0000OO)
    return OOOO0OOO000OOOO0O

def get_all_pos(input_seq):
    file_pos = open(r"/home/biocuckoo/public_html/gps-halp/code_python/GPS_pos.txt",'r')
    lines = file_pos.readlines()
    positive_seqs = []
    for line in lines:
        if ">" not in line:
            seq = line.strip()
            seq = seq.replace('U',"*")
            positive_seqs.append(seq)

    file_Blosum62 = open(r"/home/biocuckoo/public_html/gps-halp/code_python/BLOSUM62_NCBI.txt",'r')
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

        
    ###delete later
    count = [0]*21
    for i in positive_position_aa.keys():#遍历21个位置
        for key in aa:#遍历所有氨基酸
            count[i] += positive_position_aa[i][key]#计数21个位置，每个位置上总的氨基酸个数
                
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
    
    with open(r"/home/biocuckoo/public_html/gps-halp/code_python/GPS_MatrixRenew.txt") as f:
        line = f.readline()
        line = f.readline()
        weight = [float(w) for w in line.strip('\n').split('\t')]
        line = f.readline()
        line = f.readline()
        matrix_weight = [float(w) for w in line.strip('\n').split('\t')]

    train_feature_arr = encode_GPS(input_seq, train_x, train_y, positive_position_aa, BLOSUM62_dic, AA, positive_seq_num, weight, matrix_weight)
    return(train_feature_arr)


# def seq2fts(seq):
#     f1 = get_GPS(seq)
#     f2 = get_AAC(seq)
#     f3 = get_SOCNumber(seq)
#     f4 = get_PAAC(seq)
#     f5 = get_AC(seq)
#     f6 = get_binary3bit1(seq)
#     f7 = get_binary3bit5(seq)
#     f8 = get_AESNN3(seq)
#     f9 = get_PseKRAAC(seq)
#     f10 = get_GDPC(seq)
#     return [f2,f8,f5,f6,f7,f10,f1,f4,f9,f3]


def test_predict(OOO0O000OOOOO0OO0, OOO00OO00O0O0OO00):
    # feature_path = OOO0O000OOOOO0OO0
    # model_path = OOO00OO00O0O0OO00

    DNN_feature = DNN_model.predict(OOO0O000OOOOO0OO0[i])
    AE_feature = intermediate_layer_model.predict(OOO0O000OOOOO0OO0[i])

    pred_comb = comb_model.predict(test_x_Mix)
       
    return pred_comb

def out_format(pred_score, pred_type, threshold, OO0OOO0OO0OOOOO00s, ids, kn, locs, OO0OOO0OO0OOOOOOO):
    #database search
    #O0OO000OO0O000O00=files
    #O000O00O0O00OOOO0=database_sites
    #OOO00O0O0000OOO0O=lines
    #OOO0O000OOOOO0OO0=items
    O0OO000OO0O000O00 = open(r"/home/biocuckoo/public_html/gps-halp/code_python/database/verified_sites.txt",'r')
    OOO00O0O0000OOO0O = O0OO000OO0O000O00.readlines()
    O000O00O0O00OOOO0 = {}
    for line in OOO00O0O0000OOO0O:
        OOO0O000OOOOO0OO0 = line.strip("\n").split("\t")
        O000O00O0O00OOOO0[OOO0O000OOOOO0OO0[0]+"_"+OOO0O000OOOOO0OO0[1]] = OOO0O000OOOOO0OO0[2]
    O000O00O0O00OOOO0['NA'] = '-'
    O0OO000OO0O000O00.close()

    O0OO000OO0O000O00 = open(r"/home/biocuckoo/public_html/gps-halp/Basic/Protein-View-0.txt",'r')
    OOO00O0O0000OOO0O = O0OO000OO0O000O00.readlines()
    O000O00O0O00OOOO0 = {}
    for line in OOO00O0O0000OOO0O:
        OOO0O000OOOOO0OO0 = line.strip("\n").split("\t")
        O000O00O0O00OOOO0[OOO0O000OOOOO0OO0[1]] = OOO0O000OOOOO0OO0[0]
    O000O00O0O00OOOO0['NA'] = '-'
    O0OO000OO0O000O00.close()
    
    O0OO000OO0O000O00 = open(r"/home/biocuckoo/public_html/gps-halp/database/HPA_tissue_exp_site.txt",'r')
    OOO00O0O0000OOO0O = O0OO000OO0O000O00.readlines()
    O000O00O0O00OOOO0 = {}
    for line in OOO00O0O0000OOO0O:
        OOO0O000OOOOO0OO0 = line.strip("\n").split("\t")
        wrt_str = []
        for i in range(2,len(OOO0O000OOOOO0OO0),2):
            wrt_str.append(OOO0O000OOOOO0OO0[i].capitalize()+" ({0})".format(OOO0O000OOOOO0OO0[i+1]))
        O000O00O0O00OOOO0[OOO0O000OOOOO0OO0[0]] = [";".join(wrt_str),OOO0O000OOOOO0OO0[1]]
    O000O00O0O00OOOO0['NA'] = ['-','-']
    O0OO000OO0O000O00.close()

    O0OO000OO0O000O00 = open(r"/home/biocuckoo/public_html/gps-halp/code_python/database/imm_exp_uni.txt",'r')
    OOO00O0O0000OOO0O = O0OO000OO0O000O00.readlines()
    O000O00O0O00OOOO0 = {}
    for line in OOO00O0O0000OOO0O[1:]:
        OOO0O000OOOOO0OO0 = line.strip("\n").split("\t")
        wrt_str = []
        for i in range(2,len(OOO0O000OOOOO0OO0)-1):
            wrt_str.append(OOO0O000OOOOO0OO0[i].split(":")[1].replace("_"," ")+" ({0})".format(OOO0O000OOOOO0OO0[i].split(":")[0]))
        O000O00O0O00OOOO0[OOO0O000OOOOO0OO0[0]] = [";".join(wrt_str),OOO0O000OOOOO0OO0[1]]
    O000O00O0O00OOOO0['NA'] = ['-','-']
    O0OO000OO0O000O00.close()

    O0OO000OO0O000O00 = open(r"/home/biocuckoo/public_html/gps-halp/code_python/database/cancer_merge.txt",'r')
    OOO00O0O0000OOO0O = O0OO000OO0O000O00.readlines()
    O000O00O0O00OOOO0 = {}
    for line in OOO00O0O0000OOO0O[1:]:
        OOO0O000OOOOO0OO0 = line.strip("\n").split("\t")
        wrt_top = []
        wrt_str = []
        for i in range(1,len(OOO0O000OOOOO0OO0)):
            sub_items = OOO0O000OOOOO0OO0[i].split(";")
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
        O000O00O0O00OOOO0[items[0]] = ";".join(wrt_str)
    O0OO000OO0O000O00.close()
    
    out1 = open(outf, 'a+')
    size = os.path.getsize(outf)
    if size == 0:
        out1.write("ID\tPosition\tCode\tPeptide\tScore\tCutoff\tSource\tHPA Tissue\tHPA Link\tImmune Cell\tImmune Cell Link\tMutation\tUniprot\n")
    id2 = ids[0].strip()
    print("pred_type",len(pred_type),"ids",len(ids))
    for i in range(len(pred_type)):
        if pred_type[i]:
            uni_id = OO0OOO0OO0OOOOOOO[ids[i]]
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
    upfile =sys.argv[3]     #上传的fasta文件路径
    outpath =sys.argv[4]

    outf = outpath
    outpath = outpath.rsplit('.gps', 1)[0] + '_'

    GS = treeNode.split(",")
    GS = [x for x in GS]
    G0 = GS[0]

    OO0OOO0OO0OOOOOOO, OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO = fasta2seq_aa(upfile)

    data_dic = {"His":[OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO],"Arg":[OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO],"Lys":[OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO,OO0OOO0OO0OOOOOOO]}
    for nphosaa in GS:
        print(nphosaa)
        featuresall = seq2fts(data_dic[nphosaa[:-1]][1])
        knpath = "/home/biocuckoo/public_html/gps-halp/webcomp/models/AA/" + nphosaa
        pred_comb = test_predict(featuresall, knpath)
        zhibiao = pd.read_csv( knpath+ "zhibiao.txt", sep="\t", header=None)
        if threshold == "a":
            thd = "Null"
            pred_type = [1] * len(pred_comb)
            out_format(pred_comb.tolist(), pred_type, thd, data_dic[nphosaa[:-1]][1], data_dic[nphosaa[:-1]][0], nphosaa, data_dic[nphosaa[:-1]][2], OO0OOO0OO0OOOOOOO)
        else:
            thd = 0
            if threshold == "h":
                thd = round(float(zhibiao.iloc[0, 1]), 4)
            elif threshold == "m":
                thd = round(float(zhibiao.iloc[1, 1]), 4)
            elif threshold == "l":
                thd = round(float(zhibiao.iloc[2, 1]), 4)
            pred_type = (pred_comb >= thd).astype(bool)
            out_format(pred_comb.tolist(), pred_type, thd, data_dic[nphosaa[:-1]][1], data_dic[nphosaa[:-1]][0], nphosaa, data_dic[nphosaa[:-1]][2], OO0OOO0OO0OOOOOOO)
    sys.exit()

