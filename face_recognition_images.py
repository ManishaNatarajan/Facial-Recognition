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
parser.add_argument('--image', help='Image on which faces are to be recognised')
parser.add_argument('--method', help='Method for face_recognition-->hog or cnn')

args = parser.parse_args()

#Load known embeddings from dataset
print('Loading known facial embeddings')
encoded_data = pickle.loads(open(args.encodings, 'rb').read())

# Read image
image = cv2.imread(args.image)

#Change color format of images
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Get Bounding box of faces in each image
boxes = face_recognition.face_locations(image_rgb, model=args.method)

#Get Face Encodings
encodings = face_recognition.face_encodings(image_rgb, boxes)


#Initialize labels for faces detected
class_names = []

# #Go through encodings and check for matches with dataset
# for encoding in encodings:
# 	#Try to match faces in test image with known encodings
# 	matches = face_recognition.compare_faces(encoded_data['Encodings'], encoding)
# 	name = 'Anonymous'

# 	if True in matches:
# 		#Get indexes of all matched faces in image
# 		matched_indexes = [for (i, b) in enumerate(matches) if b]
# 		counts = {}
# 		#Match the face with most counts from a known face
# 		for i in matched_indexes:
# 			name = encoded_data['Class_names']

