import sys
import numpy as np
import random
import cv2
import os
import pandas as pd
import scipy
from scipy import misc
from PIL import Image

def draw_bounding_box(im, left,top,right,bottom):
    img = im
    #print left,top,right,bottom
    #rr,cc = line(int(top),int(left),int(bottom),int(right))
    #img[rr,cc,0] = 1

    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)

    color =  np.array([0,255,0],dtype=np.uint8) #Green
    img[top,left:right] = color
    img[top:bottom,left] = color
    img[bottom,left:right] = color
    img[top:bottom,right] = color

    return img 

def KL_distance(x,y):
    #x = [mean,cov]
    mean0 = x[0]
    conv0 = x[1]
    mean1 = y[0]
    conv1 = y[1]
    """
    if (mean0==mean1).all() and (conv0==conv1).all():
        return float('inf')
    """
    inv_conv1 = np.linalg.inv(conv1)
    diff = mean1 - mean0
    tr_term = np.trace(np.dot(inv_conv1,conv0))
    det_term = np.log(np.linalg.det(conv1)/np.linalg.det(conv0))
    middle_term = np.dot(np.dot(diff.T,inv_conv1),diff)
    N = mean0.shape[0]
    return 0.5 * (tr_term + middle_term + det_term - N)
  

class EM_clustering:
    def __init__(self, k, tol=0.0000000001,max_iter=100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def deltcal(self, f,g):
        # f and g list of gaussians
	delta = 0
        for fi in f:
	    distances = [KL_distance(fi,g[j]) for j in g]
	    delta += fi[2] * min(distances)
        return delta
 
    def fit(self,data):
        # change data structure in there
        self.clusters = {}
        # shuffle and randomly pick the first time
        random.shuffle(data)
        for i in range(self.k):
            self.clusters[i] = data[i]
        # here
        iter = 0
        delta = float('inf')
        delta_change = float('inf')
        while iter < self.max_iter and delta_change > self.tol:  # threshold add
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
            # E step
            for box in data:
                distances = [KL_distance(box,self.clusters[prev]) for prev in self.clusters]
		newindex = distances.index(min(distances))
                self.classifications[newindex].append(box)
            # M step
	    newclass = {}
            for i in self.classifications:
	        if not self.classifications[i]:
                    continue
                # combine gaussians together here
                beta = 0
		mean = np.zeros(2)
                for j in self.classifications[i]:
                   beta += j[2]
		   mean += j[2] * j[0]
                mean /= beta
		cov = np.matrix([[0,0],[0,0]],dtype=np.float64)
		for j in self.classifications[i]:
		   temp = (j[0] - mean)
		   temp = np.dot(temp,temp.T)
		   temp += j[1]
		   temp *= j[2]
		   cov += temp
		cov /= beta
		newgaussian = [mean, cov, beta]
		newclass[i] = newgaussian
            # update clusters as new, then calculate gain
	    new_delta = self.deltcal(data,newclass)
            #print(new_delta)
	    delta_change = delta - new_delta
	    print(iter)
            print(delta_change)
	    iter += 1
	    if delta_change > 0:
	        self.clusters = newclass
                delta = new_delta
	    
        return self.clusters

def gaussian():
    results = pd.read_csv('outbo.txt', header = None)
    data = {} # data point
    mog = {}
    getsize= {}
    getsize1 = {}
    getsize2 = {}
    weight = {} # score total
    for i in range(results.shape[0]):
        ids = results[0][i]
    #data_result.append(['image_id','x1','y1','x2','y2','type','confidence'])
    image_ids = set()
    for ids in results[0].tolist():
        image_ids.add(ids)
    for ids in image_ids:
        data[ids] = []
        getsize1[ids] = []
        getsize2[ids] = []
        weight[ids] = 0
        mog[ids] = [np.zeros(2),np.zeros((2,2))]
    for i in range(results.shape[0]):
        ids = results[0][i]
        score = float(results[6][i])
        weight[ids] += score
    for i in range(results.shape[0]):
        ids = results[0][i]
        x1 = float(results[1][i])
        y1 = float(results[2][i])
        x2 = float(results[3][i])
        y2 = float(results[4][i])
        u1 = (x2 - x1) / 2.
        u2 = (y2 - y1) / 2.
        fac = float(results[6][i]) / weight[ids]
        mean = np.asarray([u1, u2],dtype=np.float64)
        cov = np.asmatrix([[((x2 - x1) / 4.) **2, 0],[0, ((y2 - y1) / 4.) ** 2]],dtype=np.float64)
        # here, try append or sum them
        data[ids].append([mean, cov, fac])
        getsize1[ids].append(x2-x1)
        getsize2[ids].append(y2-y1)
        #mog[key][0] += fac * mean
        #mog[key][1] += fac * cov # mixture of gaussians for every image   
    for key in getsize1:
        i = 0
        sum1 = 0
        sum2 = 0
        for i in range(len(getsize1[key])):
            sum1 += getsize1[key][i]
            sum2 += getsize2[key][i]
            i += 1
        getsize1[key] = sum1 / i
        getsize2[key] = sum2 / i
    for key in getsize1:
        with Image.open(key) as img:
            width, height = img.size
            getsize[key] = int(round(width * height / (getsize1[key]*getsize2[key])))
    ans = {}
    for key in data:
        # EM here
        # randomly pick K gaussian as starting points, mog not useful here.
        # incoporate hier clustering later.
        # EM clustering:
        # model = EM_clustering(k=getsize[key])
        model = EM_clustering(k=getsize[key])
	ans[key] = model.fit(data[key])
    # drawing part
    for key in ans:
        image_name = key
        im = misc.imread(image_name)
        print(im.shape)
        right_max = 0
        bottom_max = 0
        image_height,image_width,_ = im.shape
        for indx in ans[key]:
            i = ans[key][indx]
            mean = i[0]            
            conv = i[1]
            h = np.sqrt(conv[0,0]) * 2
            w = np.sqrt(conv[1,1]) * 2
            u1 = mean[0]
            u2 = mean[1]
            x1 = u1 - w
            x2 = u1 + w
            y1 = u2 - h
            y2 = u2 + h            
            left = x1
	    top = y1
	    right = x2
	    bottom = y2
	    left = max(0,left)
	    top = max(0,top)
	    right = min(right,image_width-1)
	    bottom = min(bottom,image_height-1)
	    im = draw_bounding_box(im,left,top,right,bottom)
        misc.imsave('key.png',im)
        print(right_max,bottom_max)


if __name__ == "__main__":
    gaussian()
