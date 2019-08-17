import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob


def calibrate_camera():
	# Mapping each calibration image to number of checkerboard corners
	# Everything is (9,6) for now
	objp_dict = {
		1: (9, 5),
		2: (9, 6),
		3: (9, 6),
		4: (9, 6),
		5: (9, 5),
		6: (9, 6),
		7: (9, 6),
		8: (9, 6),
		9: (9, 6),
		10: (9, 6),
		11: (9, 6),
		12: (9, 6),
		13: (9, 6),
		14: (9, 6),
		15: (9, 6),
		16: (9, 6),
		17: (9, 6),
		18: (9, 6),
		19: (9, 6),
		20: (9, 6),
	}

	# Initialize the list of object points and corners for calibration
	objp_list = []		# 3d points in real world space
	corners_list = []	# 2d points in image plane

	# Go through all images and find corners
	for k in objp_dict:
		nx, ny = objp_dict[k]

		# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((nx*ny,3), np.float32)
		objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

		# Make a list of calibration images
		fname = 'camera_cal/calibration%s.jpg' % str(k)
		img = cv2.imread(fname)

		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		# If found, save & draw corners
		if ret == True:
			# Save object points and corresponding corners
			objp_list.append(objp)
			corners_list.append(corners)

			# Draw and display the corners
			#cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			#plt.imshow(img)
			#plt.show()
			#print('Found corners for %s' % fname)
		else:
			print('Warning: ret = %s for %s' % (ret, fname))

	# Test undistortion on an image
	img = cv2.imread('test_images/straight_lines1.jpg')
	img_size = (img.shape[1], img.shape[0])
	
	# Do camera calibration given object points and corner points
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, corners_list, img_size,None,None)

	return mtx, dist


if __name__ == '__main__':
	mtx, dist = calibrate_camera()
	save_dict = {'mtx': mtx, 'dist': dist}
	#with open('calibrate_camera.p', 'wb') as f:
	#	pickle.dump(save_dict, f)
	# or
	pickle.dump( save_dict, open( "calibrate_camera.p", "wb" ) ) 
	
	
	# For Group photos, use this part!
	# Make a list of target images
	images = glob.glob('test_images/*.jpg')

	# Step through the list of images and undistort them and then save them
	for idx, fname in enumerate(images):	
		# Undistort example calibration image
		img = mpimg.imread(fname)
		dst = cv2.undistort(img, mtx, dist, None, mtx)
		# Visualize undistortion
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
		ax1.imshow(img)
		ax1.set_title('Original Image', fontsize=20)
		ax2.imshow(dst)
		ax2.set_title('Undistorted Image', fontsize=20)
		plt.savefig('output_images/02_undistort_'+fname[12:-4]+'.png')		
		cv2.waitKey(500)
		
	cv2.destroyAllWindows()
	
	""" For Single photo, use this part!		
	k = 5 	# camera image number
	fname = 'calibration%s' % str(k)
	img = mpimg.imread('camera_cal/'+fname+'.jpg')
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	
	# Visualize undistortion
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=20)
	ax2.imshow(dst)
	ax2.set_title('Undistorted Image', fontsize=20)
	plt.savefig('output_images/02_undistort_'+fname+'.png')
	"""
	
