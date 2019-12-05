import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import pytesseract
import math
import random
from imutils.object_detection import non_max_suppression
import argparse
def Is_invertible(a):
	return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
def Solve_homography(M1,M2,M3):
	H = None
	M = np.array(
		  [[M1[0][0],M1[0][1],1,0,0,0,0,0,0],
		  [0,0,0,M1[0][0],M1[0][1],1,0,0,0],
		  [0,0,0,0,0,0,M1[0][0],M1[0][1],1],
		  [M2[0][0],M2[0][1],1,0,0,0,0,0,0],
		  [0,0,0,M2[0][0],M2[0][1],1,0,0,0],
		  [0,0,0,0,0,0,M2[0][0],M2[0][1],1],
		  [M3[0][0],M3[0][1],1,0,0,0,0,0,0],
		  [0,0,0,M3[0][0],M3[0][1],1,0,0,0],
		  [0,0,0,0,0,0,M3[0][0],M3[0][1],1]]
		)
	S = np.array([ M1[1][0],M1[1][1],1,M2[1][0],M2[1][1],1,M3[1][0],M3[1][1],1 ])
	if Is_invertible(M):
		H = np.linalg.solve(M,S)
		H = np.array([ [H[0],H[1],H[2] ],
			   		   [ H[3],H[4],H[5] ],
			   		   [ H[6],H[7],H[8] ]
			 		  ])
	return H

def Feature_matching(img1,img2):
	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	bf = cv2.BFMatcher(cv2.NORM_L2)
	matches = bf.knnMatch(des1,des2, k=2)
	good = []

	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good.append([m])

	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)
	mat = good[0][0]
	id1 = mat.queryIdx
	id2 = mat.trainIdx
	(x1,y1) = kp1[id1].pt
	List_good_point = [(kp1[mat[0].queryIdx].pt,kp2[mat[0].trainIdx].pt) for mat in good]
	return  List_good_point,img3
	
def RANSAC(List_good_point,threshold):
	Best_model = None
	N_inlier = 0
	for i in range(200):
		M1 = random.choice(List_good_point)
		M3 = M2 = M1
		while M2 == M1 or M3 == M2 or M3 == M1:
			M2 = random.choice(List_good_point)
			M3 = random.choice(List_good_point)
		H = Solve_homography(M1,M2,M3)
		if H is not None:
			count_inlier = 0
			for p in List_good_point:
				M_train = np.array([p[0][0],p[0][1],1])
				M_test = np.array([p[1][0],p[1][1],1])
				M_result = H.dot(M_train)
				d = np.sqrt((M_result[0]-M_test[0])**2+(M_result[1]-M_test[1])**2)
				if d < threshold:
					count_inlier+=1
			if count_inlier > N_inlier:
				Best_model = H
				N_inlier = count_inlier
	return Best_model

def Detection(best_model,img1,img2):
	h,w,c = img1.shape
	p1 = np.array([0,0,1])
	p2 = np.array([w,0,1])
	p3 = np.array([0,h,1])
	p4 = np.array([w,h,1])

	lt = best_model.dot(p1)
	rt = best_model.dot(p2)
	lb = best_model.dot(p3)
	rb = best_model.dot(p4)

	#cv2.line(img2,(int(lt[0]),int(lt[1])), (int(rt[0]), int(rt[1])), (0,0,255),3)
	#cv2.line(img2,(int(rt[0]),int(rt[1])), (int(rb[0]), int(rb[1])), (0,0,255),3)
	#cv2.line(img2,(int(rb[0]),int(rb[1])), (int(lb[0]), int(lb[1])), (0,0,255),3)
	#cv2.line(img2,(int(lb[0]),int(lb[1])), (int(lt[0]), int(lt[1])), (0,0,255),3)
	return lt,rt,lb,rb

def rotateImage(image, angle): 
    image_center = tuple(np.array(image.shape[1::-1])/2) 
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0) 
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255)) 
    return result 
def define_angle(x1,y1,x2,y2):
  #vector don vi ( 0,-1)
	cos_a = (y1-y2)/math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
	angle = math.acos(cos_a)*180/math.pi
	if x2 < x1:
		return -angle
	else:
		return angle
def crop_image(img_new,lt,rb,image_center,anpha):
	lt_new = rotate(lt[0],lt[1],image_center[0],image_center[1],anpha)
	rb_new = rotate(rb[0],rb[1],image_center[0],image_center[1],anpha)
	if int(rb_new[1]*1.2) < img_new.shape[1]:
		tmp = int(rb_new[1]*1.1)
	else:
		tmp = rb_new[1]
	result = img_new[lt_new[1]:tmp,lt_new[0]:rb_new[0]]
	return result
def rotate(x, y, xm, ym, a):
	a = -a * math.pi/ 180
	xr = (x - xm) * math.cos(a) - (y - ym) * math.sin(a)   + xm
	yr = (x - xm) * math.sin(a) + (y - ym) * math.cos(a)   + ym
	xr = abs(xr)
	yr = abs(yr)
	return int(xr), int(yr)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
def auto_brightness(img):
    H,W = img.shape[:2]
    brs = 0
    count =0
    #10
    for i in range(0,H,5):
        for j in range(0,W,5):
            brs+=(int(img[i,j,0])+int(img[i,j,1])+int(img[i,j,2]))/3
            count+=1
    avg=brs/count
    #160
    if(avg<160):
        v=int(160-avg)*4
        if v > 65 : v = 65
        img=increase_brightness(img,value=int(v))
    return img
def gray_scale(img):
    H,W = img.shape[:2]
    img = auto_brightness(img)  
    gray = np.zeros((H,W), np.uint8)
    for i in range(H):
        for j in range(W):
        	#1.33
            graynum=int(1.4*img[i,j,1])+abs(int(img[i,j,1])-int(img[i,j,0])) 
            if(graynum>255):
                graynum=255
            gray[i][j]=graynum
    return gray
def remove_blob(img,k):
    Y = []
    tmp,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    threshold_blobs_area = k
    for i in range(1, len(contours)):
        index_level = int(hierarchy[0][i][1])
        if index_level <= i:
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area <= threshold_blobs_area:
                Y.append(cnt[0][0][1])
                cv2.drawContours(img, [cnt], -1, 255, -1, 1)
    return img,Y
def detect_line_cmnd(img):

	h,w,k = img.shape
	b,g,r = img[:,:,0],img[:,:,1],img[:,:,2]
	h,w,k = img.shape
	new_img = np.zeros((h,w),dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			tmp = int(r[i][j]) - int(g[i][j])
			if tmp < 0 :
				tmp = 0
			new_img[i][j] = tmp
	return new_img
def find_anpha(img):
  #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #ret, img = cv2.threshold(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)
  tmp,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  rs = None
  for c in contours:
    rect = cv2.minAreaRect(c)
    x,y = rect[0][0],rect[0][1]
    h,w = rect[1][0],rect[1][1]
    if h > w :
      h,w = w,h
    x = x - w//2
    y = y - h//2
    if w > img.shape[0]//3 and y > int(img.shape[1]/9):
      rs = rect[2]
  if rs < -45:
    rs = 90 + rs    
  return rs
def bouding_box(img):
  #ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
  image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  result = None
  for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if w > img.shape[0]//3 and y > int(img.shape[1]/9) :
      result = (x,y,x+w,y+h)
    #cv2.rectangle(img, (result[0],result[1]), (result[2], result[3]), (0, 255, 0),2)
    #cv2.imshow('bouding_box',img)
  return result
def redwave_filter(img):
    H,W = img.shape[:2]
    img = auto_brightness(img)  
    gray = np.zeros((H,W), np.uint8)
    for i in range(H):
        for j in range(W):
            gray1 = int(1.45*img[i,j,2])+abs(int(img[i,j,2])-int(img[i,j,0])) #1.45
            gray2 = int(1.33*img[i,j,1])+abs(int(img[i,j,1])-int(img[i,j,0])) 
            if(gray1<gray2):
                graynum=gray2
            else:
                graynum=gray1
            if(graynum>255):
                graynum=255
                #sys.stderr.write(str(graynum))
            gray[i][j]=graynum
    return gray
def text_detection(image):
	orig = image.copy()
	(H, W) = image.shape[:2]

	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	net = cv2.dnn.readNet('frozen_east_text_detection.pb')

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)


	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.3:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	MAX_y = 0
	MIN_y = orig.shape[0]
	MAX_X = 0
	MIN_X = orig.shape[1]
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		startY = int(startY * rH)
		startX = int(startX * rW)
		endY = int(endY * rH)
		endX = int(endX * rW)

		if endY > MAX_y:
			MAX_y = endY
		if startY < MIN_y:
			MIN_y = startY
		if startX < MIN_X:
			MIN_X = startX
		if endX > MAX_X:
			MAX_X = endX
	if orig.shape[0] - MAX_y > 10:
		MAX_y += 10
	else:
		MAX_y = orig.shape[0]
	if orig.shape[1] - MAX_X > 20:
		MAX_X += 20
	else:
		MAX_X = orig.shape[1]
	if MIN_X < 15 :
		MIN_X = 0
	else:
		MIN_X = 10
	if MIN_y < 0:
		MIN_y = 0
	new_img = orig[MIN_y:MAX_y,:MAX_X]
	return new_img,MIN_y
def detect_cmnd(image):
	orig = image.copy()
	(H, W) = image.shape[:2]

	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	net = cv2.dnn.readNet('frozen_east_text_detection.pb')

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)


	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.3:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	MAX_y = 0
	MIN_y = orig.shape[0]
	MAX_X = 0
	MIN_X = orig.shape[1]
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		startY = int(startY * rH)
		startX = int(startX * rW)
		endY = int(endY * rH)
		endX = int(endX * rW)

		if endY > MAX_y:
			MAX_y = endY
		if startY < MIN_y:
			MIN_y = startY
		if startX < MIN_X:
			MIN_X = startX
		if endX > MAX_X:
			MAX_X = endX
	new_img = orig[MIN_y:MAX_y+30,MIN_X-30:]
	return new_img
def remove_name_line(img):
	img2 = img.copy()
	ret, img2 = cv2.threshold(img2, 175, 255, cv2.THRESH_BINARY_INV)
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN,kernel)
	kernel = np.ones((7,40),np.uint8)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	image, contours, hier = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		if h > w :
			h,w = w,h
		
		if x < img2.shape[1]//5 and y < int(img2.shape[0]/2) and h < int(img2.shape[0]*0.75) and w < img2.shape[1]//2 :
	  		img = cv2.rectangle(img,(0,0),(int(x+w),int(y+h)),(255,255,255),-1)
	img = cv2.resize(img,(int(img.shape[1]*1.2),int(img.shape[0]*1.2)))
	return img