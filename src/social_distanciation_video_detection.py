from bird_view_transfo_functions import compute_perspective_transform,compute_point_perspective_transformation
from tf_model_person_detection import Model 
from colors import bcolors
from functions import bird_detect_people_on_frame
import numpy as np
import itertools
import imutils
import time
import math
import glob
import yaml
import cv2
import os

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3


def get_human_box_detection(boxes,scores,height,width):
	""" 
	For each object detected, check if it is a human and if the confidence >> our threshold.
	Return 2 coordonates necessary to build the box.
	@ boxes : all our boxes coordinates
	@ scores : confidence score on how good the prediction is -> between 0 & 1
	@ classes : the class of the detected object ( 1 for human )
	@ height : of the image -> to get the real pixel value
	@ width : of the image -> to get the real pixel value
	"""
	array_boxes = list() # Create an empty list
	for i in range(boxes.shape[0]):
		# If the class of the detected object is 1 and the confidence of the prediction is > 0.6
		if scores[i] > 0.75:
			# Multiply the X coordonnate by the height of the image and the Y coordonate by the width
			# To transform the box value into pixel coordonate values.
			box = [boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3]]
			# Add the results converted to int
			array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
	return array_boxes


def get_centroids_and_groundpoints(array_boxes_detected):
	"""
	For every bounding box, compute the centroid and the point located on the bottom center of the box
	@ array_boxes_detected : list containing all our bounding boxes 
	"""
	array_centroids,array_groundpoints = [],[] # Initialize empty centroid and ground point lists 
	for index,box in enumerate(array_boxes_detected):
		# Draw the bounding box 
		# c
		# Get the both important points
		centroid,ground_point = get_points_from_box(box)
		array_centroids.append(centroid)
		array_groundpoints.append(centroid)
	return array_centroids,array_groundpoints


def get_points_from_box(box):
	"""
	Get the center of the bounding and the point "on the ground"
	@ param = box : 2 points representing the bounding box
	@ return = centroid (x1,y1) and ground point (x2,y2)
	"""
	# Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
	center_x = int(((box[1]+box[3])/2))
	center_y = int(((box[0]+box[2])/2))
	# Coordiniate on the point at the bottom center of the box
	center_y_ground = center_y + ((box[2] - box[0])/2)
	return (center_x,center_y),(center_x,int(center_y_ground))

def draw_rectangle(corner_points):
	# Draw rectangle box over the delimitation area
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)


######################################### 
# Load the config for the top-down view #
#########################################
print(bcolors.WARNING +"[ Loading config file for the bird view transformation ] "+ bcolors.ENDC)
with open("../conf/config_birdview.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0,0
corner_points = []
for section in cfg:
	corner_points.append(cfg["image_parameters"]["p1"])
	corner_points.append(cfg["image_parameters"]["p2"])
	corner_points.append(cfg["image_parameters"]["p3"])
	corner_points.append(cfg["image_parameters"]["p4"])
	width_og = int(cfg["image_parameters"]["width_og"])
	height_og = int(cfg["image_parameters"]["height_og"])
	img_path = cfg["image_parameters"]["img_path"]
	size_frame = cfg["image_parameters"]["size_frame"]
print(bcolors.OKGREEN +" Done : [ Config file loaded ] ..."+bcolors.ENDC )


######################################### 
#		     Select the model 			#
#########################################
print(bcolors.WARNING + " [ Loading the YOLO MODEL ... ]"+bcolors.ENDC)
model = Model()
print(bcolors.OKGREEN +"Done : [ Model loaded and initialized ] ..."+bcolors.ENDC)


######################################### 
#		     Select the video 			#
#########################################
video_names_list = [name for name in os.listdir("../video/") if name.endswith(".mp4") or name.endswith(".avi")]
for index,video_name in enumerate(video_names_list):
    print(" - {} [{}]".format(video_name,index))
video_num = input("Enter the exact name of the video (including .mp4 or else) : ")
if video_num == "":
	video_path="../video/PETS2009.avi"  
else :
	video_path = "../video/"+video_names_list[int(video_num)]


######################################### 
#		    Minimal distance			#
#########################################
distance_minimum = input("Prompt the size of the minimal distance between 2 pedestrians : ")
if distance_minimum == "":
	distance_minimum = "110"
distance_minimum = int(distance_minimum)

######################################################
#########									 #########
# 				START THE VIDEO STREAM               #
#########									 #########
######################################################
vs = cv2.VideoCapture(video_path)
output_video_1,output_video_2 = None,None
# Get video properties
fps = vs.get(cv2.CAP_PROP_FPS)
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
model = Model()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
if os.path.exists('../output/video.avi'):
    os.remove('../output/video.avi')
out = cv2.VideoWriter('../output/video.avi', fourcc, fps, (width, height))

# Iterate through frames
# vidlen = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    
# Loop until the end of the video stream
while vs.isOpened():	
	# Load the frame
	(frame_exists, frame) = vs.read()
	# Test if it has reached the end of the video
	if not frame_exists:
		break
	else:
		# Make the predictions for this frame
       
		frame = bird_detect_people_on_frame(frame, 0.75, distance_minimum, width, height, model)

	# Draw the green rectangle to delimitate the detection zone
	draw_rectangle(corner_points)
    
	# Show both images	
	cv2.imshow("Original picture", frame)


	key = cv2.waitKey(1) & 0xFF

	out.write(frame)

	# Break the loop
	if key == ord("q"):
		break

vs.release()
out.release()    
cv2.destroyAllWindows()
