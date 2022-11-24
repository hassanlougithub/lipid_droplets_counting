def genGausImage(framesize, mx, my, cov=1):
    x, y = np.mgrid[0:framesize, 0:framesize]
    pos = np.dstack((x, y))
    mean = [mx, my]
    cov = [[cov, 0], [0, cov]]
    rv = scipy.stats.multivariate_normal(mean, cov).pdf(pos)
    return rv/rv.sum()

def getDensity(width, markers):
    gaus_img = np.zeros((width,width))
    for k in range(width):
        for l in range(width):
            if (markers[k,l] > 0.5):
                gaus_img += genGausImage(len(markers),k-patch_size/2,l-patch_size/2,cov)
    return gaus_img

def getMarkersLipids(labelPath, scale, size):  
    labs = imread(labelPath)
    if len(labs.shape) == 2:
        lab = labs[:,:]/255
    elif len(labs.shape) == 3:
        lab = labs[:,:,0]/255
    else:
        print ("unknown label format")
    
    binsize = [scale,scale]
    out = np.zeros(size)
    for i in range(binsize[0]):
        for j in range(binsize[1]):
            out = np.maximum(lab[i::binsize[0], j::binsize[1]], out)
        
    print (lab.sum(),out.sum())
    assert np.allclose(lab.sum(),out.sum(), 1)
    
    return out

def getLipidsCounts(markers,x,y,h,w):
    types = [0] * noutputs
    for i in range(noutputs):
        types[i] = (markers[y:y+h,x:x+w] == 1).sum()
        #types[i] = (markers[y:y+h,x:x+w] != -1).sum()
    return types

def getLipidsLabels(markers, img_pad, base_x, base_y, stride, scale):
    
    height = int(((img_pad.shape[0])/stride))
    width = int(((img_pad.shape[1])/stride))
    print ("label size: ", height, width)
    labels = np.zeros((noutputs, height, width))
    if (kern == "sq"):
        for y in range(0,height):
            for x in range(0,width):
                count = getLipidsCount(markers,x*stride,y*stride,patch_size,patch_size)  
                for i in range(0,noutputs):
                    labels[i][y][x] = count[i]

    
    elif (kern == "gaus"):
        for i in range(0,noutputs):
            labels[i] = getDensity(width, markers[base_y:base_y+width,base_x:base_x+width])
    

    count_total = getLipidsCount(markers,0,0,framesize_h+patch_size,framesize_w+patch_size)
    return labels, count_total

def getTrainingExampleLipids(img_raw, framesize_w, framesize_h, labelPath, base_x,  base_y, stride, scale):
    
    img = img_raw[base_y:base_y+framesize_h, base_x:base_x+framesize_w]
    pad_width = int((patch_size)/2)
    img_pad = np.pad(img[:,:,0],pad_width, "constant")
    
    #pad_width = int((patch_size)/2)
    #img_pad = np.pad(img[:,:,0],pad_width, "constant")
    
    markers = getMarkersLipids(labelPath, scale, img_raw.shape[0:2])
    markers = markers[base_y:base_y+framesize_h, base_x:base_x+framesize_w]
    markers = np.pad(markers, patch_size, "constant", constant_values=-1)
    
    labels, count  = getLipidsLabels(markers, img_pad, base_x, base_y, stride, scale)
    return img, labels, count