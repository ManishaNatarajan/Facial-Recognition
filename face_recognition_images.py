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

print('Recognizing Faces in test image...')
#Get Bounding box of faces in each image
boxes = face_recognition.face_locations(image_rgb, model=args.method)
# print(boxes)

if(boxes == []):
	print('No faces detected. Image maybe too small')
#Get Face Encodings
encodings = face_recognition.face_encodings(image_rgb, boxes)

# print(encodings)
if(encodings == []):
	print('Could not calculate facial embeddings. Try another image')
#Initialize labels for faces detected
class_names = []

#Go through encodings and check for matches with dataset
for encoding in encodings:
	#Try to match faces in test image with known encodings
	matches = face_recognition.compare_faces(encoded_data['Encodings'], encoding)
	name = 'Muggle'

	if True in matches:
		#Get indexes of all matched faces in image
		matched_indexes = [i for (i, b) in enumerate(matches) if b]
		counts = {}
		#Match the face with most counts from a known face
		for i in matched_indexes:
			#Get the class label of the ith recognised face
			name = encoded_data['Class_names'][i]
			counts[name] = counts.get(name, 0) + 1

		#Get the max votes for each recognised face
		name =  max(counts, key= counts.get)


	#Append the class with max votes to class_names
	class_names.append(name)

#Go through each recognised face to draw bounding box and predict label
for ((top, right, bottom, left), name) in zip(boxes, class_names):
	#Draw Bounding box
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	#Place Labels appropriately
	place = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, place), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

print(class_names)

#Display the image
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()





