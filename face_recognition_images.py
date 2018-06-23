#To generate Facial Embeddings of images in the dataset
#Created by Manisha

#Import required Libraries
import cv2
import os
import argparse
import pickle
import face_recognition

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to folder containing Images')
parser.add_argument('--encoding', help='Path to folder for storing facial encodings')
parser.add_argument('--method', help='method for detecting faces in the image')

args = vars(parser.parse_args())

#Get list of Image Paths
imagePaths = []
for dir in os.listdir(args["dataset"]):
	sub_dir = os.path.join(args['dataset'], dir)
	print(sub_dir)
	listOfFiles =  os.listdir(sub_dir) 
	for file in listOfFiles:
		file = os.path.join(sub_dir, file)
		imagePaths.append((os.path.abspath(file)))

# print(imagePaths)

get_encodings=[]
get_class_names=[]

#Go through the entire dataset
for (i, path) in enumerate(imagePaths):
	#Print Log Info..
	print('Encoding Image:{}/{}'.format(i+1, len(imagePaths)))
	
	#Get names of classes of each image -> from sub directory
	class_name = path.split(os.path.sep)[-2]

	# Read image
	image = cv2.imread(path)

	#Change color format of images
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	#Get Bounding box of faces in each image
	boxes = face_recognition.face_locations(image_rgb, model=args['method'])

	#Get Face Encodings
	encodings = face_recognition.face_encodings(image_rgb, boxes)

	#Make list of Encodings
	for encoding in encodings:
		get_encodings.append(encoding)
		get_class_names.append(class_name)

# print(get_class_names)

#Write the encodings to disk
print("Writing Encodings")
data = {'Encodings': get_encodings, 'Class_names': get_class_names}
f = open(args['encoding'], 'wb')
f.write(pickle.dumps(data))
f.close()	



