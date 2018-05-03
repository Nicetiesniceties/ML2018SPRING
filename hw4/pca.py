import sys
import os
import numpy as np
from skimage import io

def data_processing(vec, syntaxerr, reshape):
	temp_vec = vec
	if(syntaxerr):
		vec -= np.min(vec)
		vec /= np.max(vec)
		vec = (vec * 255).astype(np.uint8)
	if(reshape):
		temp_vec = temp_vec.reshape(600, 600, 3)
	return temp_vec

if __name__ == '__main__':
	img_names = os.listdir(sys.argv[1])
	N = len(img_names)
	X = []
	print("----reading all the faces----")
	for i in range(N):
		img = io.imread(os.path.join(sys.argv[1], img_names[i]))
		X.append(img.flatten())
	print("----SVDing----")
	X = np.array(X)
	X = X.T
	X_mean = np.mean(X, axis = 1)
	print("X.size = " + str(X.shape))
	X_eg_vecs, s, V = np.linalg.svd(X - X_mean[:, None], full_matrices = False)
	
	target_img = io.imread(os.path.join(sys.argv[1], sys.argv[2]))
	target_img = target_img.flatten()
	target_img = np.array(target_img)
	print("----start painting----")
	print("----rebuild face----\n")
	eg_face = X_eg_vecs[:, 0:4]
	original_img = target_img
	reconstruct_img = np.zeros(X_mean.shape)
	for i in range(4):
		x_weight = np.dot((original_img - X_mean).T, eg_face[:, i])
		u = eg_face[:, i]
		reconstruct_img += x_weight * u
	reconstruct_img +=  X_mean
	reconstruct_img = data_processing(reconstruct_img, 1, 1)
	io.imsave('./reconstruction.png', reconstruct_img)
