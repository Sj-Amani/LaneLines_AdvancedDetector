import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	"""
	Takes an image, gradient orientation, and threshold min/max values
	"""
	# Apply the following steps to img
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 2) Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
	if   orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	elif orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	
	# 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	
	# 4) Create a mask of 1's where the scaled gradient magnitude is >= thresh_min and <= thresh_max
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# 5) Return this mask as the result
	return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
	"""
	Return the magnitude of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Apply the following steps to img
    
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	
	# 3) Calculate the gradient magnitude
	mag_sobel = np.sqrt(sobelx**2 + sobely**2)
	
	# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
	scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
	
	# 5) Create a binary mask where mag thresholds are met, zeros otherwise
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= mag_thresh[0] ) & (scaled_sobel <= mag_thresh[1])] = 1

	# 6) Return this mask as your binary_output image
	return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
	Return the direction of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Apply the following steps to img
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	
	# 3) Take the absolute value of the x and y gradients
	#    and use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
	dir_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

	# 4) Create a binary mask where direction thresholds are met
	binary_output = np.zeros_like(gray)
	binary_output[(dir_sobel >= thresh[0] ) & (dir_sobel <= thresh[1])] = 1

	# 5) Return this mask as your binary_output image
	return binary_output


def hls_thresh(img, thresh=(100, 255)):
	"""
	Convert RGB to HLS and threshold to binary image using S channel
	"""
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output


def hsv_thresh(img, hsv_rangeLower = np.array([10, 100, 100]), hsv_rangeUpper = np.array([22, 230, 255])):
	"""
	Convert RGB to HSV and threshold to binary image
	I used this for detecting the YELLOW lines on the road
	"""
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv_temp = cv2.inRange(hsv, hsv_rangeLower, hsv_rangeUpper)
	binary_output = np.zeros_like(hsv_temp)
	binary_output[(hsv_temp != 0)] = 1
	return binary_output

def gray_thresh(img, thresh = [202]):
	"""
	Convert RGB to GRAY and threshold to binary image
	I used it for detecting the WHITE lines on the road
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	binary_output = np.zeros_like(gray)
	binary_output[(gray > thresh)] = 1
	return binary_output


def combined_thresh(img):
	#abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255)
	abs_bin  = abs_sobel_thresh(img, orient='x', thresh_min=5, thresh_max=100)
	#mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
	mag_bin  = mag_thresh(img, sobel_kernel=5, mag_thresh=(5, 100))
	dir_bin  = dir_threshold(img, sobel_kernel=7, thresh=(0.84, 1.117))
	hls_bin  = hls_thresh(img, thresh=(170, 255))
	hsv_bin  = hsv_thresh(img, np.array([10, 62, 100]), np.array([22, 230, 255]))
	gray_bin = gray_thresh(img, thresh = [150])

	shape_bin = np.zeros_like(dir_bin)
	color_bin = np.zeros_like(dir_bin)
	combined = np.zeros_like(dir_bin)

	### combination method 1
	#mag_dir_bin = cv2.bitwise_or(mag_bin, dir_bin)
	#mag_dir_gray_bin = cv2.bitwise_or(mag_dir_bin, gray_bin)
	#combined[(((abs_bin == 1 | ( mag_dir_gray_bin == 1)) | hls_bin == 1) | hsv_bin != 0) ] = 1
	
	### combination method 2
	#combined[(((abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1) | hsv_bin != 0) | gray_bin == 1 ] = 1
	
	### combination method 3
	#First, detect the lane line shape areas 
	#shape_bin[(abs_bin == 1 | ((mag_bin == 1) | (dir_bin == 1)))] = 1
	#shape_bin = dir_bin
	#Second, detect the lane line color areas
	#color_bin[(hsv_bin != 0 | ((hls_bin == 1) | (gray_bin == 1)))] = 1
	#Third, combine first and second
	#combined[(shape_bin == 1) & (color_bin == 1)] = 1
	
	### combination method 4
	#First, detect the lane line shape areas 
	shape_bin[( (dir_bin == 1) & ((abs_bin == 1) | (mag_bin == 1)) )] = 1
	#Second, detect the lane line color areas
	color_bin[( (hsv_bin == 1) | ((hls_bin == 1) | (gray_bin == 1)) )] = 1
	#Third, combine first and second
	combined[((shape_bin == 1) & (color_bin == 1))] = 1
	#combined = cv2.bitwise_and(shape_bin, color_bin)
	return combined, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin, gray_bin  # DEBUG


if __name__ == '__main__':
	#img_file = 'test_images/straight_lines2.jpg'
	#img_file = 'test_images/test5.jpg' 
	img_file = 'test_images/For_Challenge/challenge_video04.jpg'
	#img_file = 'test_images/For_Challenge/harder_challenge_video01.jpg'

	with open('calibrate_camera.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']

	img = mpimg.imread(img_file)
	img = cv2.undistort(img, mtx, dist, None, mtx)

	combined, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin, gray_bin = combined_thresh(img)

	plt.subplot(3, 3, 1)
	plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 2)
	plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 3)
	plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 4)
	plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 5)
	plt.imshow(img)
	plt.subplot(3, 3, 6)
	plt.imshow(combined, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 7)
	plt.imshow(hsv_bin, cmap='gray', vmin=0, vmax=1)
	plt.subplot(3, 3, 9)
	plt.imshow(gray_bin, cmap='gray', vmin=0, vmax=1)

	plt.tight_layout()
	plt.show()
