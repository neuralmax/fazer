import pygame.camera
import pygame.image
import sys
import cv2
import numpy as np
from PIL import Image
def convert_to_vector(image):
	# Read the input image
	#image = cv2.imread(input_image_path)
	# Convert the image to grayscale
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Apply Canny edge detection to find outlines
	edges = cv2.Canny(gray_image, 100, 200)
	# Find contours (shapes) in the image
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Create a new blank image to draw the vectorized image
	vector_image = np.zeros_like(image)
	# Draw contours on the blank image to form vectorized shapes
	cv2.drawContours(vector_image, contours, -1, (255, 255, 255), 1)
	return vector_image,contours
	# Save the vectorized image
	#cv2.imwrite(output_image_path, vector_image)
def synth(frequency=100, duration=1.5, sampling_rate=44100):
	frames = int(duration*sampling_rate)
	arr = np.cos(2*np.pi*frequency*np.linspace(0,duration, frames))
	arr = np.clip(arr*10, -1, 1) # squarish waves
	fade = list(np.ones(frames-4410))+list(np.linspace(1, 0, 4410))
	arr = np.multiply(arr, np.asarray(fade))
	sound = np.asarray([32767*arr,32767*arr]).T.astype(np.int16)
	sound = pygame.sndarray.make_sound(sound.copy())
	return sound
def vectosound(cnt):
	duration=0.1
	sampling_rate=44100
	#frequency=40
	frequency=cnt%1000
	frames = int(duration*sampling_rate)
	arrl = np.cos(2*np.pi*frequency*np.linspace(0,duration, frames))
	#arrl = np.clip(arrl*10, -1, 1) # squarish waves
	arrr = np.sin(2*np.pi*frequency*np.linspace(0,duration, frames))
	#arrr = np.clip(arrr*10, -1, 1)
	fade = list(np.ones(frames-4410))+list(np.linspace(1, 0, 4410))
	arrl = np.multiply(arrl, np.asarray(fade))
	arrr = np.multiply(arrr, np.asarray(fade))
	sound = np.asarray([32767*arrl,32767*arrr]).T.astype(np.int16)
	sound = pygame.sndarray.make_sound(sound.copy())
	return sound
pygame.camera.init()
pygame.mixer.init()
cameras = pygame.camera.list_cameras()
print("Using camera %s ..." % cameras[0])
webcam = pygame.camera.Camera(cameras[0])
webcam.start()
# grab first frame
img = webcam.get_image()
WIDTH = img.get_width()
HEIGHT = img.get_height()
screen = pygame.display.set_mode( ( WIDTH, HEIGHT ) )
pygame.display.set_caption("pyGame Camera View")
counter=0
while True :
	for e in pygame.event.get() :
		if e.type == pygame.QUIT :
			sys.exit()
	counter+=1
	# draw frame
	# grab next frame
	img = webcam.get_image()
	#pil_image = PIL.Image.open(img).convert('RGB')
	data = pygame.image.tostring(img,'RGB')

	pil_image = Image.frombytes("RGB",(WIDTH,HEIGHT),data)


	open_cv_image = np.array(pil_image)
	# Convert RGB to BGR
	open_cv_image = open_cv_image[:, :, ::-1].copy()
	imgv,vec=convert_to_vector(open_cv_image)
	print(vec[0])
	pgimgv=pygame.image.frombuffer(imgv.tostring(), imgv.shape[1::-1], "BGR")
	screen.blit(pgimgv, (0,0))
	vectosound(counter).play()
	pygame.display.flip()
