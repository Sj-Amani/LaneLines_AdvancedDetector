import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
import glob

def perspective_transform(img):
	"""
	Execute perspective transform
	"""
	img_size = (img.shape[1], img.shape[0])

	src = np.float32(
		[[200, 720],
		[1100, 720],
		[595, 450],
		[685, 450]])
	dst = np.float32(
		[[300, 720],
		[980, 720],
		[300, 0],
		[980, 0]])

	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
	unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

	return warped, unwarped, m, m_inv


if __name__ == '__main__':
	

	with open('calibrate_camera.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']

	
	# For Group photos, use this part!
	# Make a list of target images
	images = glob.glob('test_images/*.jpg')

	# Step through the list of images and undistort them and then save them
	for idx, fname in enumerate(images):	
		# Undistort example calibration image
		img = mpimg.imread(fname)
		dst = cv2.undistort(img, mtx, dist, None, mtx)
		combined, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin, gray_bin = combined_thresh(dst)
		warped, unwarped, m, m_inv = perspective_transform(combined)
		# Visualize undistortion
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
		ax1.imshow(img)
		ax1.set_title('Original Image', fontsize=20)
		ax2.imshow(warped, cmap='gray', vmin=0, vmax=1)
		ax2.set_title('Warped Image', fontsize=20)
		plt.savefig('output_images/04_warped_'+fname[12:-4]+'.png')
		plt.show()		
		cv2.waitKey(500)
		cv2.destroyAllWindows()
	
	""" 
	
	# For Single photo, use this part!
	img_file = 'test_images/test5.jpg'
	img = mpimg.imread(img_file)
	img = cv2.undistort(img, mtx, dist, None, mtx)

	img, abs_bin, mag_bin, dir_bin, hls_bin, hsv_bin, gray_bin = combined_thresh(img)

	warped, unwarped, m, m_inv = perspective_transform(img)

	plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
	plt.show()

	plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
	plt.show()
	"""
