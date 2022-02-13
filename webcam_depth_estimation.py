import cv2
import numpy as np

from packnetsfm import PackNetSfM

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize model
max_dist = 3.0
model_path='models/resnet18_mr_selfsup_d_384x640.onnx'
depth_estimator = PackNetSfM(model_path)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Read frame from the video
	ret, frame = cap.read()
	if not ret:	
		break
	
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

