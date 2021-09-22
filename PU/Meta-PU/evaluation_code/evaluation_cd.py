import os
import argparse
import numpy as np

from pyemd import emd_samples
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors


def evaluation(file_name):
    gt_path = os.path.join(args.gt_path, file_name)
    pre_path = os.path.join(args.pre_path, file_name)
    assert os.path.exists(gt_path)
    assert os.path.exists(pre_path)

    gt_points = np.loadtxt(gt_path)
    pre_points = np.loadtxt(pre_path)

    gt2pre, _ = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pre_points).kneighbors(gt_points)
    pre2gt, _ = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gt_points).kneighbors(pre_points)

    return np.squeeze(gt2pre), np.squeeze(pre2gt), emd_samples(gt_points, pre_points)


parser = argparse.ArgumentParser()
parser.add_argument('--pre_path', required=True)
parser.add_argument('--gt_path',  required=True)
args = parser.parse_args()

files = [f for f in os.listdir(args.gt_path) if f.endswith('.xyz')]
distances = Pool(32).map(evaluation, files)
gt2pre, pre2gt, emd = zip(*distances)
gt2pre, pre2gt = np.hstack(gt2pre), np.hstack(pre2gt)
print('GT  --> PRE')
print('\tMean     : {}'.format(np.mean(gt2pre)))
print('\tStd      : {}'.format(np.std(gt2pre)))
print('\tRecall   : {}'.format(np.mean(gt2pre <= 1e-2)))
print('\tRecall   : {}'.format(np.mean(gt2pre <= 2e-2)))
print('PRE --> GT')
print('\tMean     : {}'.format(np.mean(pre2gt)))
print('\tStd      : {}'.format(np.std(pre2gt)))
print('\tPrecision: {}'.format(np.mean(pre2gt <= 1e-2)))
print('\tPrecision: {}'.format(np.mean(pre2gt <= 2e-2)))
print('CD:')
print('\t{}'.format(0.5*(np.mean(gt2pre)+np.mean(pre2gt))))
print('F-score:')
print('\t{}'.format(2/(1/np.mean(gt2pre <= 1e-2)+1/np.mean(pre2gt <= 1e-2))))
print('\t{}'.format(2/(1/np.mean(gt2pre <= 2e-2)+1/np.mean(pre2gt <= 2e-2))))
print('EMD:')
print('\t{}'.format(np.mean(emd)))
