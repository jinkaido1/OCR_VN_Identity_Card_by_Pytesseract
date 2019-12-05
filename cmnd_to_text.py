import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import pytesseract
from skimage.filters import threshold_yen,threshold_otsu
import matplotlib.cm as cm
from Read_text import *
if __name__ == '__main__':
	#Read train data
	img1 = cv2.imread('cmndqt.jpg')
	#Read file test
	img2 = cv2.imread('test_2.jpg')
	#--------------------------------------------------------
	#Detect cmnd
	List_good_point,img3 = Feature_matching(img1,img2)
	best_model = RANSAC(List_good_point,1)
	lt,rt,lb,rb = Detection(best_model,img1,img2)
	#img3 = img2.copy()
	#cv2.line(img3,(int(lt[0]),int(lt[1])), (int(rt[0]), int(rt[1])), (0,0,255),3)
	#cv2.line(img3,(int(rt[0]),int(rt[1])), (int(rb[0]), int(rb[1])), (0,0,255),3)
	#cv2.line(img3,(int(rb[0]),int(rb[1])), (int(lb[0]), int(lb[1])), (0,0,255),3)
	#cv2.line(img3,(int(lb[0]),int(lb[1])), (int(lt[0]), int(lt[1])), (0,0,255),3)
	#cv2.imwrite('detect_cmnd_image.jpg',img3)

	## File angle and rotate image
	anpha = define_angle(lb[0],lb[1],lt[0],lt[1])
	img_rotate = rotateImage(img2,anpha)
	image_center = tuple(np.array(img2.shape[1::-1])/2)
	rs = crop_image(img_rotate,lt,rb,image_center,anpha)
	#cv2.imwrite('rs.jpg',rs)
	#---------------------------------------------------------
	top_end_text = detect_cmnd(rs)
	top_end_text = cv2.resize(top_end_text,(994,908))
	#cv2.imwrite('top_end_text.jpg',top_end_text)
	new_img = detect_line_cmnd(top_end_text)

	#cv2.imwrite('anhxam.jpg',new_img)
	ret2,th2 = cv2.threshold(new_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#opening image to conect character in a word
	# open (3,3) , closing (3,27)
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN,kernel)
	#Closing image to conect word in a line
	kernel = np.ones((3,39),np.uint8)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	#red_img = cv2.resize(closing, (1000,635), interpolation = cv2.INTER_AREA)
	red_img = closing
	#cv2.imwrite('closing.jpg',closing)
	#Boungding box
	angle=find_anpha(red_img)
	img = rotateImage(red_img,angle)
	#plt.imsave('red.jpg',img,cmap =cm.gray)
	result = bouding_box(img)
	#cv2.rectangle(img, (result[0],result[1]), (result[2], result[3]), (0, 255, 0),2)
	#cv2.imshow('bounding_box',img)
	#rs = cv2.resize(rs, (1000,635), interpolation = cv2.INTER_AREA)
	top_end_text = rotateImage(top_end_text,angle)
	'''
	w = result[2] - result[0]
	x = result[0] - w//6.5

	ALL = rs[result[3]:,int(x):result[2]+int(w//(8.45))
	'''
	ALL = top_end_text[result[3]:,:]
	#cv2.imwrite('pre-ALL.jpg',ALL)
	#--------------------------------------------------------
	ALL,MIN_Y = text_detection(ALL)
	#cv2.imwrite('ALL.jpg',ALL)
	img = gray_scale(ALL)
	#cv2.imwrite('gray_image.jpg',img)
	ret, img = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#cv2.imwrite('bw_image.jpg',img)
	kernel = np.ones((3,5),np.uint8)
	opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
	#cv2.imwrite('opening.jpg',opening)
	erosion,Y = remove_blob(opening,70)
	#cv2.imwrite('erosion.jpg',erosion)
	#--------------------------------------------------------
	#erosion = cv2.imread('erosion.jpg')
	#new_img,MIN_Y = text_detection(erosion)
	new_img = erosion
	h = new_img.shape[0]
	NAME = new_img[int(41*h/417):int(167*h/417),:]   
	#NAME =cv2.rectangle(NAME,(0,0),(int(NAME.shape[1]*0.22),NAME.shape[0]//2),(255,255,255),-1)
	#NAME = cv2.resize(NAME,(NAME.shape[1]*2,NAME.shape[0]*2))
	DAY = new_img[int(157*h/417):int(213*h/417),int(new_img.shape[1]/3.5):int(new_img.shape[1]*0.9)]
	PLACE = new_img[int(213*h/417):int(315*h/417),:]
	#PLACE = cv2.rectangle(PLACE,(0,0),(int(PLACE.shape[1]*0.35),int(PLACE.shape[0]*0.7)),(255,255,255),-1)
	DKHK = new_img[int(315*h/417):]
	#DKHK = cv2.resize(DKHK,(DKHK.shape[1]*2,DKHK.shape[0]*2))
	#DKHK = cv2.rectangle(DKHK,(0,0),(int(DKHK.shape[1]*0.6),int(DKHK.shape[0]/2.4)),(255,255,255),-1)
	ID = ALL[:MIN_Y+int(45*h/417),int(new_img.shape[1]*0.3):int(new_img.shape[1]*0.9)] 
	ID = redwave_filter(ID)
	ret2,ID = cv2.threshold(ID,175,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	ID,Y = remove_blob(ID,70)
	ID = cv2.medianBlur(ID,5)
	ID =cv2.resize(ID,(ID.shape[1]*2,ID.shape[0]*2))
	#cv2.imwrite('ID.jpg',ID)
	cv2.imwrite('NAME.jpg',NAME)
	s_ID =  pytesseract.image_to_string(ID ,lang='eng',config='--psm 10  --oem 3 -c tessedit_char_whitelist=0123456789')
	s_Day = pytesseract.image_to_string(DAY,lang='vie2',config='--psm 6')
	L = [NAME,PLACE,DKHK]
	Result = [s_ID,s_Day]


	for i,vl in enumerate(L):
		kq = remove_name_line(vl)
		cv2.imwrite('Name_line_'+str(i)+'.jpg',kq)
		kq = cv2.rectangle(kq,(0,0),(int(kq.shape[1]/5),int(kq.shape[0]/0.4)),(255,255,255),-1)
		s = pytesseract.image_to_string(kq,lang='vie2',config='--psm 6')
		l_check = ['\n',' ','a','ă', 'â', 'b' ,'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 
					'o','ô', 'ơ', 'p', 'q', 'r','s', 't', 'u', 'ư', 'v', 'x', 'y','0','1','2','3','4','5','6','7','8','9',
					'á','à','ã','ả','ạ','ấ','ầ','ẫ','ả','ậ','ắ','ằ','ẵ','ẳ','ặ',
					'é','è','ẽ','ẻ','ẹ','ế','ề','ễ','ể','ệ','í','ì','ĩ','ỉ','ị',
					'ó','ò','õ','ỏ','ọ','ố','ồ','ỗ','ổ','ộ','ớ','ờ','ỡ','ở','ợ',
					'ú','ù','ũ','ủ','ụ','ứ','ừ','ữ','ử','ự','ý','ỳ','ỹ','ỷ','ỵ',
					'z','w','f','Z','W','F']
		s = s.lower()
		print(s)
		for c in s:
			if c not in l_check:
				s =s.translate({ord(c): None})
		Result.append(s)
	print(Result)