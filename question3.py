import cv2
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FOLDER = Path('a2images')

# Step 1: Load the images
arch_image_path = DATA_FOLDER/'005.jpg'  # Replace with your architectural image
flag_image_path = DATA_FOLDER/'flag.png'  # Replace with your flag image

arch_image = cv2.imread(arch_image_path)
flag_image = cv2.imread(flag_image_path)

# Step 2: Define a function to select points on the architectural image
points = []  # List to store the selected points

def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(arch_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", arch_image)

# Step 3: Select 4 points on the planar surface
arch_image_rgb = cv2.cvtColor(arch_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib
fig, ax = plt.subplots()

plt.imshow(arch_image_rgb)
plt.title("Select 4 Points")
plt.axis('off')

# Use ginput to select 4 points
points = plt.ginput(4)  # Wait until 4 points are selected
plt.show()

# Step 4: Get the dimensions of the flag image
height, width, _ = flag_image.shape

# Step 5: Define the 4 corner points of the flag image (source points)
src_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

# Step 6: Convert selected points from the architectural image to numpy array (destination points)
dst_points = np.array(points, dtype=np.float32)

# Step 7: Compute the homography matrix
homography_matrix, _ = cv2.findHomography(src_points, dst_points)

# Step 8: Warp the flag image using the homography matrix
warped_flag = cv2.warpPerspective(flag_image, homography_matrix, (arch_image.shape[1], arch_image.shape[0]))

# Step 9: Create a mask of the warped flag
gray_warped_flag = cv2.cvtColor(warped_flag, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_warped_flag, 1, 255, cv2.THRESH_BINARY)

# Step 10: Add transparency to the flag
alpha = 0.5  # Transparency level (0.0 to 1.0)
warped_flag = cv2.addWeighted(warped_flag, alpha, arch_image, 1 - alpha, 0)

# Step 11: Blend the warped flag onto the architectural image
mask_inv = cv2.bitwise_not(mask)
arch_image_bg = cv2.bitwise_and(arch_image, arch_image, mask=mask_inv)
flag_fg = cv2.bitwise_and(warped_flag, warped_flag, mask=mask)

result_image = cv2.add(arch_image_bg, flag_fg)

# Step 12: Display the result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis(False)
plt.show()