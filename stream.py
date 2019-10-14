import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
#=======================================================
use_gpu = torch.cuda.is_available()

if use_gpu:
	print('USING CUDA')
	device = torch.device('cuda:0')
else:
	print('USING CPU')
	device = torch.device('cpu')

#model structure loaded
model = models.vgg16_bn(pretrained = True)
#freezing the weights
for param in model.parameters():
	param.requires_grad = False

#modifying the last layer
num_in = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # take all but last layer
features.extend([nn.Linear(num_in,7)])#custom last layer
model.classifier = nn.Sequential(*features)# attaching it back to classifier

model.to(device)
model.eval()
#load state_dict saved
model.load_state_dict(torch.load('checkpoint.pt'))




#======================================================

def get_label(img):

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)
	loader = transforms.Compose([transforms.Resize(224),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[.229, 0.224, 0.225])])

	img = loader(img).float()
	img = img.unsqueeze(0)
	print(img.size())
	img = img.to(device)
	output = model(img)


	prediction = output.data.max(1, keepdim = True)[1]

	return prediction

def draw_box(img,prev_label,test):
	names =['Angry','Disgust','Fear',' Happy','Neutral','Sad','Surprise']
	
	haar_data ='haar_data/haarcascade_frontalface_default.xml'
	cascade = cv2.CascadeClassifier(haar_data)

	faces = cascade.detectMultiScale(img,minNeighbors = 5)
	# print(len(faces))
	bound_img = img
	pre_label = prev_label
	for f in faces:
		x,y,w,h = f
		pass_img = bound_img[y:y+h,x:x+w]
		label  = get_label(pass_img)
		print(label.item())
		bound_img = cv2.rectangle(bound_img, (x, y), (x + w, y + h), (0,255,0), 1)

		if test%3 ==0:

			cv2.putText(bound_img, names[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
			pre_label = names[label]
		else:

			cv2.putText(bound_img, prev_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
			pre_label = prev_label



	return bound_img,pre_label

def stream():

	cap = cv2.VideoCapture(0)
	prev_label = 'Neutral'
	i = 0
	while (cap.isOpened()):

		ret,frame = cap.read()
		
		

		img,prev_label = draw_box(frame,prev_label,i)

		if i %3	 ==0:
			i =0

		cv2.imshow('Result',img)
		i +=1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':

	stream()
