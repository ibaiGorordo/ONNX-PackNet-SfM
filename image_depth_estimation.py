import cv2
import numpy as np
from imread_from_url import imread_from_url

from packnetsfm import PackNetSfM

# Initialize model
max_dist =5.0
model_path='models/resnet18_mr_selfsup_d_384x640.onnx'
depth_estimator = PackNetSfM(model_path)

# Read inference image
img = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/HydeStreetSF.JPG/1280px-HydeStreetSF.JPG")

# Estimate depth and colorize it
depth_map = depth_estimator(img)
color_depth = depth_estimator.draw_depth(max_dist)

combined_img = np.hstack((img, color_depth))

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
cv2.imshow("Estimated depth", combined_img)
cv2.waitKey(0)