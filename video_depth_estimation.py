import cv2
import pafy
import numpy as np

from packnetsfm import PackNetSfM

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

videoUrl = 'https://youtu.be/z_9GiRz12-4'
start_time = 6 # skip first {start_time} seconds
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Initialize model
max_dist = 5.0
model_path='models/resnet18_mr_selfsup_d_384x640.onnx'
depth_estimator = PackNetSfM(model_path)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	try:
		# Read frame from the video
		ret, frame = cap.read()
		if not ret:	
			break
	except:
		continue
	
	# Estimate depth and colorize it
	depth_map = depth_estimator(frame)
	color_depth = depth_estimator.draw_depth(max_dist)

	combined_img = np.hstack((frame, color_depth))

	cv2.imshow("Estimated depth", combined_img)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
