from __future__ import print_function

import numpy

from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.audioSegmentation import smoothMovingAvg

""" Utility function for counting silence periods """


def silenceCounter(x, fs, st_win, st_step, smoothWindow=0.5, weight=0.5, plot=False):
	if weight >= 1:
		weight = 0.99
	if weight <= 0:
		weight = 0.01

	# Step 1: feature extraction
	x = audioBasicIO.stereo2mono(x)
	st_feats, _ = aF.stFeatureExtraction(x, fs, st_win * fs,
										st_step * fs)

	# Step 2: train binary svm classifier of low vs high energy frames
	# keep only the energy short-term sequence (2nd feature)
	st_energy = st_feats[1, :]
	en = numpy.sort(st_energy)
	# number of 10% of the total short-term windows
	l1 = int(len(en) / 10)
	# compute "lower" 10% energy threshold
	t1 = numpy.mean(en[0:l1]) + 0.000000000000001
	# compute "higher" 10% energy threshold
	t2 = numpy.mean(en[-l1:-1]) + 0.000000000000001
	# get all features that correspond to low energy
	class1 = st_feats[:, numpy.where(st_energy <= t1)[0]]
	# get all features that correspond to high energy
	class2 = st_feats[:, numpy.where(st_energy >= t2)[0]]
	# form the binary classification task and ...
	
	# change the order of the array
	# faets_s = [class1.T, class2.T]
	
	# changing order gives the segmens with silence
	faets_s = [class2.T, class1.T]
	
	# normalize and train the respective svm probabilistic model
	# (SILENCE vs ONSET)
	[faets_s_norm, means_s, stds_s] = aT.normalizeFeatures(faets_s)
	svm = aT.trainSVM(faets_s_norm, 1.0)

	# Step 3: compute onset probability based on the trained svm
	prob_on_set = []
	for i in range(st_feats.shape[1]):
		# for each frame
		cur_fv = (st_feats[:, i] - means_s) / stds_s
		# get svm probability (that it belongs to the ONSET class)
		prob_on_set.append(svm.predict_proba(cur_fv.reshape(1, -1))[0][1])
	prob_on_set = numpy.array(prob_on_set)
	# smooth probability:
	prob_on_set = smoothMovingAvg(prob_on_set, smoothWindow / st_step)

	# Step 4A: detect onset frame indices:
	prog_on_set_sort = numpy.sort(prob_on_set)
	# find probability Threshold as a weighted average
	# of top 10% and lower 10% of the values
	Nt = int(prog_on_set_sort.shape[0] / 10)
	T = (numpy.mean((1 - weight) * prog_on_set_sort[0:Nt]) + weight * numpy.mean(
		prog_on_set_sort[-Nt::]))

	max_idx = numpy.where(prob_on_set > T)[0]
	# get the indices of the frames that satisfy the thresholding
	i = 0
	time_clusters = []
	seg_limits = []

	# Step 4B: group frame indices to onset segments
	while i < len(max_idx):
		# for each of the detected onset indices
		cur_cluster = [max_idx[i]]
		if i == len(max_idx) - 1:
			break
		while max_idx[i + 1] - cur_cluster[-1] <= 2:
			cur_cluster.append(max_idx[i + 1])
			i += 1
			if i == len(max_idx) - 1:
				break
		i += 1
		time_clusters.append(cur_cluster)
		seg_limits.append([cur_cluster[0] * st_step,
							cur_cluster[-1] * st_step])

	# Step 5: Post process: remove very small segments:
	min_dur = 0.2
	seg_limits_2 = []
	for s in seg_limits:
		if s[1] - s[0] > min_dur:
			seg_limits_2.append(s)
	print(f"SEGMENTS 0.2: {seg_limits_2}")
	print(F"SEGMENTS: {seg_limits}")
