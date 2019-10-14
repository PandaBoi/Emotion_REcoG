import pandas as pd
import numpy as np
import cv2
import pickle
import glob
import sys
import os
sys.path.append('../')



#=========================================================================================================
'''
labels

0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

'''
def load_img_fer(path):
	siz = (48,48)
	df = pd.read_csv(path)
	imgs_buf = df['pixels']
	emo = df['emotion']
	imgs =[]
	emotions = []
	for im in imgs_buf:

		temp = im.split(' ')
		temp = [int(t) for t in temp]
		temp = np.array(temp,'float32').reshape(48,48)

		face = cv2.resize(temp.astype('uint8'),siz)
		imgs.append(face.astype('float32'))

	return imgs , emo



#one time saving purpose to reduce time of running=================
# imgs , emotion = load_img_fer('fer2013.csv')
# with open('images.pkl','wb') as f:
# 	pickle.dump(imgs, f)
# 	f.close()
# with open('emotions.pkl','wb') as t:
# 	pickle.dump(emotion, t)
# 	t.close()

# print('done saving')
#====================================================================




#Debugging purposes=================
def debug(imgs,emotion,idx = 15):
	names =['Angry','Disgust','Fear',' Happy','Sad','Surprise','Neutral']
	print(names[emotion[idx]])

	# cv2.namedWindow(emotion[0])
	temp = cv2.resize(imgs[idx].astype('uint8'),(48,48))
	cv2.imshow(names[emotion[idx]],temp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#====================================================================================================================




def crop_face(img):

	haar_d = 'haar_data/haarcascade_frontalface_default.xml'
	cascade = cv2.CascadeClassifier(haar_d)

	faces = cascade.detectMultiScale(img)
	if len(faces) ==0:
		print("error:No face")
	for f in faces:
		x,y,w,h = f
		sub_face = img[y:y+h,x:x+w]
		sub_face = cv2.resize(sub_face,(48,48))
		return sub_face



def load_img_jaffe(dir_path):

	labels = ['ANGRY','DISGUST','FEAR','HAPPY','SAD','SURPRISE','NEUTRAL']

	imgs =[]
	emo =[]

	for idx,label in enumerate(labels):

		lis = glob.glob(dir_path+'/'+label+'/*')
		for l in lis:
			print('in')
			img = cv2.imread(l,0)
			
			imgs.append(crop_face(img).astype('float32'))
			emo.append(idx)

	return imgs , emo








#appending to existing data===================================================================

# images,emos = load_img_jaffe(os.getcwd()+'/jaffe')
# images =np.array(images)
# emos = np.array(emos)


# with open('images.pkl','rb') as f:
# 	img_buff = pickle.load(f)
# 	f.close()

# with open('emotions.pkl','rb') as t:
# 	emo_buff = pickle.load(t)
# 	t.close()

# img_buff = np.concatenate([img_buff,images])
# emo_buff = np.concatenate([emo_buff,emos])
# # print(np.shape(emo_buff))
# # print(np.shape(img_buff))


# with open('images.pkl','wb') as f:
# 	pickle.dump(img_buff, f)
# 	f.close()
# with open('emotions.pkl','wb') as t:
# 	pickle.dump(emo_buff, t)
# 	t.close()

# print('done saving')

#====================================================================
#converting number to labels--ONE TIME ONLY
# with open('images.pkl','rb') as f:
# 	img_buff = pickle.load(f)
# 	f.close()

# with open('emotions.pkl','rb') as t:
# 	emo_buff = pickle.load(t)
# 	t.close()

# # debug(img_buff, emo_buff,36000)
# names =['Angry','Disgust','Fear',' Happy','Sad','Surprise','Neutral']
# emot_buff =[]
# for idx,e in enumerate(emo_buff):
# 	emot_buff.append(names[e])
# # print(emot_buff)

# with open('emotions.pkl','wb') as t:
# 	pickle.dump(emot_buff,t)
# 	t.close()
#-----------------------------------------

#pre-proccess FERG data================================================================

def load_ferg(path,out_path = None):


	folders = glob.glob(path+'/*')
	# print(folders)
	sub_f = []
	for f in folders:
		sub_f.append(glob.glob(f+'/*'))
	# print(sub_f)
	dic = {'sadness':'SAD',
			'disgust':'DISGUST',
			'anger':'ANGRY',
			'neutral':'NEUTRAL',
			'fear':'FEAR',
			'joy':'HAPPY',
			'surprise':'SURPRISE'}
	for a in sub_f:
		for b in a:
			name = b.split('_')[-1]
			lis = glob.glob(b+'/*.png')
			# print(lis)
			for l in lis:
				img = cv2.imread(l)
				im_name = l.split('/')[-1]
				print(out_path+dic[name])
				cv2.imwrite(out_path+dic[name]+'/'+im_name,img)




# load_ferg('FERG_DB_256','test2/')