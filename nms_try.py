import numpy as np
import numpy as np
import pandas as pd
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


def median(l):
    l = sorted(l)
    l_len = len(l)
    if l_len < 1:
        return None
    if l_len % 2 == 0 :
	return ( l[(l_len-1)/2] + l[(l_len+1)/2] ) / 2.0
    else:
	return l[(l_len-1)/2]

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    group = {}
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        temp = np.where(ovr > thresh)[0]
        group[i] = temp
        '''
        curr = dets[temp]
        #order = np.delete(order, temp)
        # may not be max for score
        a = np.median(curr[:,0])
        b = np.median(curr[:,1])
        c = np.median(curr[:,2])
        d = np.median(curr[:,3])
        e = np.median(curr[:,4])
        keep.append([a,b,c,d,e])
        '''
	inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep, group 


results = pd.read_csv('outbo.txt', header=None)
data = {}
image_ids = set()
for ids in results[0].tolist():
    image_ids.add(ids)
for ids in image_ids:
    data[ids] = []
for i in range(results.shape[0]):
    ids = results[0][i]
    x1 = float(results[1][i])
    y1 = float(results[2][i])
    x2 = float(results[3][i])
    y2 = float(results[4][i])
    score = float(results[6][i])
    #if score > 0.5: 
    data[ids].append(np.array([x1,y1,x2,y2,score]))
det_data = data
'''
#drawing
n = 0
for key in data:
    n += 1
    image_name = key
    im = misc.imread(image_name)
    for i in data[key]:
        im = draw_bounding_box(im,i[0],i[1],i[2],i[3])
    misc.imsave(str(n) + 'demo_res.jpg',im)

# det_data = data
det_data = {}
for ids in data:
    # convert list to numpy array
    temp = data[ids][0]
    for i in data[ids]:
          temp = np.vstack((temp,i))
    # 0.3 later
    curr, group = py_cpu_nms(temp, 0.5)
    det_data[ids] = []
    for i in curr:
        a = []
        b = []
        c = [] 
        d = [] 
        e = []
        a.append(temp[i][0])
        b.append(temp[i][1])
	c.append(temp[i][2])
	d.append(temp[i][3])
	e.append(temp[i][4])
        for j in group[i]:
	    a.append(temp[j][0])
	    b.append(temp[j][1])
	    c.append(temp[j][2])
	    d.append(temp[j][3])
	    e.append(temp[j][4])
        det_data[ids].append([median(a),median(b),median(c),median(d),max(e)])
'''
filename = '/home/griffin/tf-faster-rcnn/demo_results.txt'
with open(filename,'wt') as f:
    for ids in det_data:
        lst = det_data[ids]
        for i in range(len(lst)):
            temp = lst[i]
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(ids,temp[4],temp[0],temp[1],temp[2],temp[3]))

