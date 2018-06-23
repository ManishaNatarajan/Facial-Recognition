#Facial Recognition in Images using dlib
#Created by Manisha

#Import required libraries
import cv2
import argparse
import face_recognition
import pickle


#Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--encodings', help='Path to encodings file')
parser.add_arguments('--image', help='Image on which faces are to be recognised')
parser.add_arguments('--method', help='Method for face_recognition-->hog or cnn')

args = parser.parse_args()

#Load known embeddings from dataset
print('[INFO]: Loading known facial embeddings')
encoded_data = pickle.loads(open(args.encodings, 'rb').read())

#Load the image and convert to RGB
image = cv2.imread(args.image)
image_rbg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Detect bounding boxes of recognised faces in images
print('[INFO]: Recognizing Faces...')
boxes = face_recognition.face_locations(images_rgb, args.method)
encodings = face_recognition(image_rgb, boxes)

#Initialize labels for faces detected
class_names = []

#Go through encodings and check for matches with dataset
for encoding in encodings:
	