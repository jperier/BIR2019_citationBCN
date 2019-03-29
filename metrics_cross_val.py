import os
import shutil
import json

SER_DIR = 'ser_sen'

with open('metrics_cross_val.txt','w') as f:
    liste = os.listdir(SER_DIR)
    liste.sort()

    macrof_scores = []
    f_scores = []
    for i in range(6):
        f_scores.append([])

    for d in liste:
        
        if (len(d)<5 and len(macrof_scores)>0):
            for i,t in enumerate(f_scores):
                if not len(t) == 0:
                    f.write('F1_'+str(i)+': '+str(sum(t)/len(t))+'\n')
            f.write('mean: ' + str(sum(macrof_scores)/len(macrof_scores)) + '\n\n')
            macrof_scores = []
            f_scores = []
            for i in range(6):
                f_scores.append([])
        else:
            liste2 = os.listdir(SER_DIR + '/' + d)
            liste2.sort()
            max = 0
            max_epoch = 0
            max_f_scores = []
            for i in range(6):
                max_f_scores.append([])
            for m in liste2:
                if 'metrics' in m:
                    with open((SER_DIR + '/' + d + '/' + m), 'r') as mfile:
                        metrics = json.load(mfile)
                        macrof = metrics['validation_macro-f']
                        if macrof > max:
                            max = macrof
                            max_epoch = metrics['epoch']
                            for i in range(6):         
                                max_f_scores[i] = metrics[('validation_f1_' + str(i))]
                            
            macrof_scores.append(max)
            for i in range(6):
                if max_f_scores[i] != []:
                    f_scores[i].append(max_f_scores[i])

    for i,t in enumerate(f_scores):
        if not len(t) == 0:
            f.write('F1_'+str(i)+': '+str(sum(t)/len(t))+'\n')
            
    f.write('mean: ' + str(sum(macrof_scores)/len(macrof_scores)) + '\n\n')

