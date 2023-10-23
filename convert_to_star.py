import numpy as np 
import glob 
import os
import pandas as pd 

all_scores = glob.glob('/hpc/home/qh36/research/qh36/3D_picking/joint_denoising_detection/new_10215_new_eval/00024-eval-ssdn-gaussian-iter250k-0.95-0.05-joint/eval_imgs/*.txt')
with open('10215_nms20_05_095_full.star','w') as f:
    f.write('# version 30001\n\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnMicrographName #3\n_rlnAutopickFigureOfMerit #4\n')
    for sc in all_scores:
        name = os.path.basename(sc)
        name = name[:-18]
        name = name + '.mrc'
        # print(name)
        coords1 = pd.read_csv(sc, sep='\t')
        thres = 0.13
        for x, y, s in zip(coords1.x_coord, coords1.y_coord, coords1.score):
            if s > thres:
                if x > 15 and x < 1425 and y > 15 and y < 1009:
                    f.write(str(int((x)*4)) + '\t' + str(int((y)*4)) + '\t' + name + '\t' + str(s) + '\n')