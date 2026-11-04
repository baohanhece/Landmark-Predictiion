import os
import sys
import numpy as np
import cv2
import csv
import tqdm
import imutils
import torch
import time
import pandas as pd
from skimage import transform
from skimage import io as sk_io
from PIL import Image
from pandas.io.parsers import read_csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from PySide2 import QtCore, QtGui, QtWidgets
import time
#import progressbar
from time import sleep
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session

#print(tf.__version__)

from os import path
QtCore.QCoreApplication.addLibraryPath(path.join(path.dirname(QtCore.__file__), "plugins"))
QtGui.QImageReader.supportedImageFormats()

from utils.model import load_model
from utils.cldetection_utils import check_and_make_dir, calculate_prediction_metrics, visualize_prediction_landmarks


NUM_LM = 17



SIZE_IMAGE = 400#400
SIZE_SUB_IMAGE = 250
SIZE_SUB_IMAGE1 = 200
SIZE_SUB_IMAGE2 = 150
SIZE_SUB_IMAGE3 = 100
SIZE_SUB_IMAGE4 = 50

test_csv_path = './test1.csv'
save_image =True
save_image_dir = './visualize/'




# data parameters | 数据文件路径
test_csv_path = './test1.csv'

# model load dir path | 存放模型的文件夹路径
load_weight_path = './model/best_model.pt'


# model hyper-parameters: image_width and image_height
image_width = 512
image_height = 512

# model training hyper-parameters
cuda_id = 1

model_name ='UNet'

def heatmaps_to_landmarks(heatmaps):
    """
    heatmaps: NumPy array with shape (19, H, W)
    returns: NumPy array with shape (19, 2), in (x, y) order
    """
    num_landmarks, height, width = heatmaps.shape
    landmarks = np.zeros((num_landmarks, 2), dtype=np.float32)

    for i in range(num_landmarks):
        y, x = np.unravel_index(
            np.argmax(heatmaps[i]),
            heatmaps[i].shape
        )
        landmarks[i] = [x, y]

    return landmarks

def load_model_and_predict(name_of_model, X_test):


    K.clear_session()
    #print(name_of_model+"_lite.tflite",X_test.shape)
    interpreter = tf.lite.Interpreter(model_path=name_of_model+"_lite.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    #print(interpreter.get_input_details())
    output_details = interpreter.get_output_details()

    # Test model on some input data.
    input_shape = input_details[0]['shape']


    input_data = X_test.reshape(input_shape)
    #print("load_model_and_predict",X_test.shape, input_data.shape)
    interpreter.set_tensor(input_details[0]['index'], X_test)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_predict = [0] * (len(output_data[0]))
    #print (len(output_data[0]),len(y_predict))

    for num_p2 in range (len(output_data[0])-1):
	    #print (y_test[num_p1][num_p2] , end=" ")
	    y_predict[num_p2] = ((output_data[0][num_p2] ))#* (SIZE_IMAGE/2)) + (SIZE_IMAGE/2))
	    #fp.write(str(y_predict)+",")

    y_predict[num_p2+1] = ((output_data[0][num_p2+1])) #* (SIZE_IMAGE/2)) + (SIZE_IMAGE/2))

    return y_predict



def loadtocsv(blank_image,h,w,size_img ):
	#K.clear_session()
	h, w = blank_image.shape
	array_try =[[0 for x in range(h*w)] for y in range(1)] 
	kl = 0
	for j in range (0,h):
		for mn in range (0,w):
			array_try[0][kl]=(blank_image[j,mn]/ 255.)
			kl = kl + 1
	
	array_try = np.array(array_try, dtype=np.float32)
	X = array_try.reshape(-1,size_img,size_img,1)
	#K.clear_session()
	return X

def model1 (IMG_rgb_original,name_model):
	global over
	over = 0
	#K.clear_session()

	h, w,a = IMG_rgb_original.shape
	IMG_rgb_original = cv2.cvtColor(IMG_rgb_original, cv2.COLOR_BGR2GRAY)
	ratio = h/SIZE_IMAGE
	
	if h < w:

		dis = w - h
		IMG_rgb1 = np.zeros((h,h), np.uint8)
		IMG_rgb1[0:h, 0:h]= IMG_rgb_original[0:h, dis:w] #img_test_400
		IMG_rgb = cv2.resize(IMG_rgb1,(int(h/ratio), int(h/ratio))) 
	else:
	
		IMG_rgb = cv2.resize(IMG_rgb_original,(int(w/ratio), int(h/ratio)))

	#print (IMG_rgb.shape)
	blank_image = np.zeros((SIZE_IMAGE,SIZE_IMAGE), np.uint8)
	h1, w1 = IMG_rgb.shape
	blank_image[0:IMG_rgb.shape[0], 0:IMG_rgb.shape[1]]= IMG_rgb

	

	X_test = loadtocsv(blank_image,h,w,SIZE_IMAGE)
	
	#print ("../models_quanti/"+name_model)	
	y_test = load_model_and_predict("/home/jolie/Desktop/3471833/src/ex1/AI_program/inner line/models/"+name_model,X_test)

	


	#y_predict =(y_test** (SIZE_IMAGE/2)) + (SIZE_IMAGE/2)
	y_predict = [(x * (SIZE_IMAGE/2))+ (SIZE_IMAGE/2) for x in y_test]
	#print(y_predict,y_test)
	#print (y_predict)

	return (y_predict)
	
'''
def predict_try (IMG_rgb,x_point,y_point,SIZE_AREA,k,h,verOrHor):

	#K.clear_session()		
	y_1 = int(y_point- SIZE_AREA/2)
	if int(y_point+ SIZE_AREA/2) > h:
		y_2 = h
		if y_2 - y_1 < SIZE_AREA:
			y_1 = y_2 - SIZE_AREA
			#print("speciallllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")

	else:
		y_2 = int(y_point+ SIZE_AREA/2) 
	
	x_1 = int(x_point - SIZE_AREA/2)
	x_2 = int(x_point + SIZE_AREA/2)

	
	IMG_rgb2= IMG_rgb[y_1:y_2,x_1: x_2]
		
	X_test = loadtocsv(IMG_rgb2,SIZE_AREA,SIZE_AREA,SIZE_AREA)
	#print ("X_test",X_test.shape)
	
	if verOrHor == 0: #vertical
		y_test = load_model_and_predict("./csv_vert_quanti/"+str(SIZE_AREA)+ "model_point"+ str(int(k/2)),X_test)
	else: #horizontal
		y_test = load_model_and_predict("./csv_hori_quanti/"+str(SIZE_AREA)+ "model_point"+ str(int(k/2)),X_test)
	#y_test = model4.predict()
	#K.clear_session()
	return y_test,x_1,y_1
'''

def predict_try (IMG_rgb2,x_1,y_1,SIZE_AREA,k,h,verOrHor):

	
	
	X_test = loadtocsv(IMG_rgb2,SIZE_AREA,SIZE_AREA,SIZE_AREA)

	if verOrHor == 0: #vertical
		y_test = load_model_and_predict("/home/jolie/Desktop/3471833/src/ex1/AI_program/inner line/csv_vert_quanti/"+str(SIZE_AREA)+ "model_point"+ str(int(k/2)),X_test)
	else: #horizontal
		y_test = load_model_and_predict("/home/jolie/Desktop/3471833/src/ex1/AI_program/inner line/csv_hori_quanti/"+str(SIZE_AREA)+ "model_point"+ str(int(k/2)),X_test)
	#print(y_test)        
	#y_test = model4.predict()
	#K.clear_session()
	return y_test,x_1,y_1



def check_area_small (IMG_rgb,x_point,y_point,SIZE_AREA,k,h,verOrHor):

	#K.clear_session()		
	y_1 = int(y_point- SIZE_AREA/2)
	if int(y_point+ SIZE_AREA/2) > h:
		y_2 = h
		if y_2 - y_1 < SIZE_AREA:
			y_1 = y_2 - SIZE_AREA
			#print("speciallllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")

	else:
		y_2 = int(y_point+ SIZE_AREA/2) 
	
	x_1 = int(x_point - SIZE_AREA/2)
	x_2 = int(x_point + SIZE_AREA/2)

	
	IMG_rgb2= IMG_rgb[y_1:y_2,x_1: x_2]
	h1, w1 = IMG_rgb2.shape
	#print("IMG_rgb2.shape",h1, w1)
	if h1* w1 < SIZE_AREA*SIZE_AREA	:
		return 0,0,0
	else:
		return IMG_rgb2,x_1,y_1


def test_point(IMG_rgb_original,y_predict_array):
	
	K.clear_session()
	IMG_rgb = cv2.cvtColor(IMG_rgb_original, cv2.COLOR_BGR2GRAY)

	h, w = IMG_rgb.shape	
	if h > w:
		verOrHor = 0
		dis = 0
	else:
		verOrHor = 1
		dis = w - h

	y_predict_last = [0] * NUM_LM*2

	num_y_predict_last = 0

	
	for k in range (0, 25*2,2):#NUM_LM*
		
		if k == 16 or k == 20 or k == 26 or k == 36 or k == 38 or k == 44 or k == 46 or k == 48:
			continue
		
		x_point = float(y_predict_array[k])*(h/SIZE_IMAGE)+dis
		y_point = float(y_predict_array[k + 1])*(h/SIZE_IMAGE)
	
		IMG_rgb2,x_from_check,y_from_check = check_area_small (IMG_rgb,x_point,y_point,SIZE_SUB_IMAGE,k,h,verOrHor)
		if  x_from_check != 0:
			y_test,x_1,y_1 = predict_try (IMG_rgb2,x_from_check,y_from_check,SIZE_SUB_IMAGE,k,h,verOrHor)		
			x_predict = (y_test[0] * (SIZE_SUB_IMAGE/2)) + (SIZE_SUB_IMAGE/2)
			y_predict = (y_test[1] * (SIZE_SUB_IMAGE/2)) + (SIZE_SUB_IMAGE/2)
			#print (x_predict, y_predict)

			
			IMG_rgb2,x_from_check,y_from_check = check_area_small (IMG_rgb,x_predict+x_1,y_predict+y_1,SIZE_SUB_IMAGE1,k,h,verOrHor)
			if  x_from_check != 0:
				y_test2,x_1,y_1 = predict_try (IMG_rgb2,x_from_check,y_from_check,SIZE_SUB_IMAGE1,k,h,verOrHor)
				x_predict = (y_test2[0] * (SIZE_SUB_IMAGE1/2)) + (SIZE_SUB_IMAGE1/2)
				y_predict = (y_test2[1] * (SIZE_SUB_IMAGE1/2)) + (SIZE_SUB_IMAGE1/2)
				#print (x_predict, y_predict)
				#print ("2")			

				

				IMG_rgb2,x_from_check,y_from_check = check_area_small (IMG_rgb,x_predict+x_1,y_predict+y_1,SIZE_SUB_IMAGE2,k,h,verOrHor)
				if x_from_check != 0:
					y_test2,x_1,y_1 = predict_try (IMG_rgb2,x_from_check,y_from_check,SIZE_SUB_IMAGE2,k,h,verOrHor)			
					x_predict = (y_test2[0] * (SIZE_SUB_IMAGE2/2)) + (SIZE_SUB_IMAGE2/2)
					y_predict = (y_test2[1] * (SIZE_SUB_IMAGE2/2)) + (SIZE_SUB_IMAGE2/2)
					#print (x_predict, y_predict)
					#print ("3")	

					
		
					IMG_rgb2,x_from_check,y_from_check = check_area_small (IMG_rgb,x_predict+x_1,y_predict+y_1,SIZE_SUB_IMAGE3,k,h,verOrHor)
					if x_from_check != 0:
						y_test2,x_1,y_1 = predict_try (IMG_rgb2,x_from_check,y_from_check,SIZE_SUB_IMAGE3,k,h,verOrHor)	
						x_predict = (y_test2[0] * (SIZE_SUB_IMAGE3/2)) + (SIZE_SUB_IMAGE3/2)
						y_predict = (y_test2[1] * (SIZE_SUB_IMAGE3/2)) + (SIZE_SUB_IMAGE3/2)	
						#print (x_predict, y_predict)
						#print ("4")
				    
						

						IMG_rgb2,x_from_check,y_from_check = check_area_small (IMG_rgb,x_predict+x_1,y_predict+y_1,SIZE_SUB_IMAGE4,k,h,verOrHor)
						if x_from_check != 0:
							y_test2,x_1,y_1 = predict_try (IMG_rgb2,x_from_check,y_from_check,SIZE_SUB_IMAGE4,k,h,verOrHor)
							x_predict = (y_test2[0] * (SIZE_SUB_IMAGE4/2)) + (SIZE_SUB_IMAGE4/2)
							y_predict = (y_test2[1] * (SIZE_SUB_IMAGE4/2)) + (SIZE_SUB_IMAGE4/2)					
		
 		
		
			y_predict_last[num_y_predict_last] = x_predict+x_1
			y_predict_last[num_y_predict_last+1] = y_predict+y_1
			num_y_predict_last = num_y_predict_last + 2
		else:
			y_predict_last[num_y_predict_last] = x_point
			y_predict_last[num_y_predict_last+1] = y_point
			num_y_predict_last = num_y_predict_last + 2
		

		
	return y_predict_last

def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def save_predict_landmarks_to_csv(test_result_dict, output_csv):
    """
    Save image filename and predicted landmark coordinates to CSV.

    Columns:
        file_name, p1_x, p1_y, p2_x, p2_y, ...

    Uses 'predict_all' to save all model predictions.
    """
    rows = []

    for image_path, result in test_result_dict.items():
        points = np.asarray(result['predict_Unet']).reshape(-1, 2)

        row = {
            'file_name': os.path.basename(image_path)
        }

        for i, (x, y) in enumerate(points, start=1):
            row['p{}_x'.format(i)] = float(x)
            row['p{}_y'.format(i)] = float(y)

        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("Saved predicted landmarks to:", output_csv)


def main(name_file):
	global over
	t0 = time.time()
	K.clear_session()
	if (is_ascii(name_file) == False):
	    im_pil = Image.open(name_file).convert('RGB')        
	    open_cv_image = np.array(im_pil)
	    IMG_rgb_original = open_cv_image [:,:,::-1].copy()
	    #cv2.imshow('image',IMG_rgb_original)
	    #cv2.WaitKey(0)
        
	else:
	    IMG_rgb_original = cv2.imread(name_file)
	
	#IMG_rgb2 = cv2.imread("./original/3.jpg")
	h, w,a = IMG_rgb_original.shape
	if h > w:
		y_predict_25LM = model1 (IMG_rgb_original,"model2_400_vert")
	else:
		y_predict_25LM = model1 (IMG_rgb_original,"model2_400_hori")
        
	y_predict_17LM = test_point(IMG_rgb_original,y_predict_25LM)        
	t1 = time.time()
	total = t1-t0

	print("Total time taken for all process of 1 images (minutes): " + str(total/60))        

	'''
	landmark= [1502.102930098772, 987.4589773416519, 774.2524394309148, 1078.9319464191794, 486.42840130627155, 1217.0747554898262, 1347.12811845541, 1220.2745090052485, 1502.592980882153, 1527.1911668777466, 985.9797816164792, 1540.2042679786682, 1473.4659916982055, 1627.652750160545, 1372.580181133002, 2059.9895850867033, 1378.615618005395, 2225.6872534304857, 1224.2708785682917, 2235.3509395718575, 657.5729297846556, 1931.9764435216784, 1450.9975657463074, 1936.65358774364, 1474.023125834763, 1697.700372427702, 1346.35181504488, 1844.6156231500208, 1263.5080504715443, 1897.426319912076, 1477.7275944799185, 1857.0688603371382, 1029.7472193986177, 1761.9989813901484]
	for i in range (0, len(landmark),2):#NUM_LM*
		cv2.circle(IMG_rgb_original,(int(landmark[i]),int(landmark[i+1])),5,(0,255,255),-1 )
		cv2.circle(IMG_rgb_original,(int(y_predict_17LM[i]),int(y_predict_17LM[i+1])),5,(0,0,255),-1 )
	cv2.imwrite("./image/a.png" ,IMG_rgb_original)
	'''
	#print (y_predict_17LM)
	return y_predict_17LM

	'''
	####edit size of image to be 2500,2048#####
	if (h < 2500):
	    IMG_rgb1 = np.zeros((2500,2048), np.uint8)
	    IMG_rgb = cv2.cvtColor(IMG_rgb_original, cv2.COLOR_BGR2GRAY)
	    IMG_rgb2 = cv2.cvtColor(IMG_rgb2, cv2.COLOR_BGR2GRAY)  
	    IMG_rgb1[(2500-h):2500, 0:IMG_rgb1.shape[1]]= IMG_rgb
	    IMG_rgb1[0:(2500-h+30), 0:IMG_rgb1.shape[1]]= IMG_rgb2[0:(2500-h+30), 0:IMG_rgb2.shape[1]]


	for k in range (0, NUM_LM*2,2):#NUM_LM*

		y_predict_25LM[k] = float(y_predict_25LM[k])*(h/SIZE_IMAGE)
		y_predict_25LM[k + 1] = float(y_predict_25LM[k + 1])*(h/SIZE_IMAGE)



		print(k)
		#cv2.circle(IMG_rgb1,(int(y_predict_25LM[k]),int(y_predict_25LM[k+1])), 5, (255,0,0), -1)

	#cv2.imwrite("./image/b.png" ,IMG_rgb1)
        '''

	'''
	#### cut image to fix the ratio 400*400 ####
	if over != 0:
	    print (over*h/SIZE_IMAGE)
	    IMG_rgb = cv2.cvtColor(IMG_rgb_original, cv2.COLOR_BGR2GRAY)
	    blank_image = np.zeros((h,w-int(over*h/SIZE_IMAGE)), np.uint8)
	    blank_image[0:IMG_rgb.shape[0], 0:(IMG_rgb.shape[1]-int(over*h/SIZE_IMAGE))]= IMG_rgb[0:IMG_rgb.shape[0], int(over*h/SIZE_IMAGE):IMG_rgb.shape[1]]	

	    for k in range (0, NUM_LM*2,2):
	        cv2.circle(blank_image,(int(y_predict_25LM[k]),int(y_predict_25LM[k+1])), 5, (255,0,0), -1)

	    cv2.imwrite("./image/a.png" ,blank_image)
	    y_predict_25LM[k] = y_predict_25LM[k] + int(over*h/SIZE_IMAGE)
	'''   
def swap_landmark_order(landmarks):
    order = [1,0,3,2,6,7,8,9,10,12,11,5,4]
    #reverse_order = np.argsort(order)
    return landmarks[order]



gpu_id = cuda_id
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

model = load_model(model_name=model_name)
model.load_state_dict(
    torch.load(load_weight_path, map_location=device)
)
model = model.to(device)


# load test.csv
df = pd.read_csv(test_csv_path)

# test result dict
test_result_dict = {}
predict_landmarks_all = []
scale= 0.1
# test all images
with torch.no_grad():
        model.eval()
        for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            image_file_path = str(df.iloc[index, 0])
            landmarks = df.iloc[index, 2:].values.astype('float')
            landmarks = landmarks.reshape(-1, 2)
            
            # Remove landmarks 6, 10, 13, 14, 15, 16 (1-based numbering)
            remove_indices = np.array([8,12, 13, 14, 15,18])

            landmarks = np.delete(landmarks, remove_indices, axis=0)

            y_predict_17LM = main(image_file_path)
            predict_landmarks = np.asarray(y_predict_17LM).reshape(-1, 2)

            remove_indices = np.array([12,14,15,16])
            predict_landmarks = np.delete(predict_landmarks, remove_indices, axis=0)
            reorder_landmark = swap_landmark_order(predict_landmarks)
            #print(predict_landmarks,reorder_landmark)



            # load image array
            image = sk_io.imread(image_file_path)
            h, w = image.shape[:2]
            new_h, new_w = image_height, image_width

            # preprocessing image for model input
            image = transform.resize(image, (new_h, new_w), mode='constant', preserve_range=False)
            transpose_image = np.transpose(image, (2, 0, 1))
            torch_image = torch.from_numpy(transpose_image[np.newaxis, :, :, :]).float().to(device)

            # predict heatmap
            heatmap = model(torch_image)
            
            # transfer to landmarks
            heatmap = np.squeeze(heatmap.cpu().numpy())
            predict_landmarks = []
            predict_landmarks_all = []
            for i in range(np.shape(heatmap)[0]):
                landmark_heatmap = heatmap[i, :, :]
                yy, xx = np.where(landmark_heatmap == np.max(landmark_heatmap))
                # there may be multiple maximum positions, and a simple average is performed as the final result
                x0, y0 = np.mean(xx), np.mean(yy)
                # zoom to original image size
                x0, y0 = x0 * w / new_w, y0 * h / new_h
                # append to predict landmarks
                #if i in range(0,19):
                predict_landmarks.append([x0, y0])
                #predict_landmarks_all.append([x0, y0])

                #print (predict_landmarks,predict_landmarks_all)

            remove_indices = np.array([8,12, 13, 14, 15,18])
            predict_Unet = np.delete(predict_landmarks, remove_indices, axis=0)	



            #print (predict_landmarks,predict_landmarks_all)
            test_result_dict[image_file_path] = {
	            'scale': scale,
	            'gt': np.asarray(landmarks),
	            'predict_our': np.asarray(reorder_landmark),
                'predict_Unet': np.asarray(predict_Unet)
                }
            #break


print("Number of test cases:", len(test_result_dict))

if len(test_result_dict) > 0:
    first_key = next(iter(test_result_dict))
    first_result = test_result_dict[first_key]

    print("GT shape:", first_result['gt'].shape)
    print("Predict shape:", first_result['predict_our'].shape)
    print("Predict shape:", first_result['predict_Unet'].shape)

    total_gt_landmarks = sum(
        result['gt'].shape[0]
        for result in test_result_dict.values()
    )

    total_our_predict_landmarks = sum(
        result['predict_our'].shape[0]
        for result in test_result_dict.values()
    )

    total_UNET_predict_landmarks = sum(
        result['predict_Unet'].shape[0]
        for result in test_result_dict.values()
    )

    print("Total GT landmarks:", total_gt_landmarks)
    print("Total our predicted landmarks:", total_our_predict_landmarks)
    print("Total UNET predicted landmarks:", total_UNET_predict_landmarks)
calculate_prediction_metrics(test_result_dict)

# Skip metric calculation because GT has 19 points
# but the model predicts 38 points.
# calculate_prediction_metrics(test_result_dict)


if save_image:
    
    check_and_make_dir(save_image_dir)
    visualize_prediction_landmarks(
        test_result_dict,
        save_image_dir
    )

    save_predict_landmarks_to_csv(test_result_dict,'./full38baseline.csv')







