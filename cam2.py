import pygame.camera
import pygame.image
import sys
import cv2
import numpy as np
from PIL import Image
import time
fps=1
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
def flatten(vec):
	#vec[0][0][0][0]
	lens=[]
	#startingpnts=len(vec)*2
	samplingrate=44100
	totalpnts=samplingrate//fps
	pnts=[]
	for line in vec:
		first=True
		linevecs=np.diff(line,axis=0)
		lens.append(np.linalg.norm(linevecs,axis=1).sum())
		pnts.append(len(line))
	startingpnts=sum(pnts)
	samplestodo=totalpnts-startingpnts
	totallen=sum(lens)
	samplelinelen=totallen/samplestodo
	arrl=[]
	arrr=[]
	for line,ln in zip(vec,lens):
		#print('ln//samplelinelen',ln//samplelinelen)
		xtodo=np.linspace(0.0,10.0,abs(int(ln//samplelinelen))+2)
		linet=np.transpose(line)
		xdata=linet[0][0]
		ydata=linet[1][0]
		xcurrent=np.linspace(0.0,10.0,len(xdata))
		#print('xtodo',xtodo)
		#print('xcurrent',xcurrent)
		#print('xdata',xdata)
		xinterp=np.interp(xtodo,xcurrent,xdata)
		yinterp=np.interp(xtodo,xcurrent,ydata)

		arrl=np.append(arrl,xinterp)
		arrr=np.append(arrr,yinterp)
		arrls = (arrl-np.min(arrl))/(np.max(arrl)-np.min(arrl))*2-1
		arrrs = (arrr-np.min(arrr))/(np.max(arrr)-np.min(arrr))*-2-1
		#arrr+=list(yinterp)
	print('arrl len',sum(arrl))
	return [arrrs,arrls],[arrl,arrr]
def vectosound(vec):
	vecr,vecl=vec
	duration=1/fps
	sampling_rate=44100
	#frequency=40
	#frequency=cnt%1000
	sound = np.asarray([32767*vecl,32767*vecr]).T.astype(np.int16)
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
	#checkpoint=pygame.time.Clock.tick(10)
	time.sleep(1/fps)
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
	print(vec[0][0][0][0])
	pgimgv=pygame.image.frombuffer(imgv.tostring(), imgv.shape[1::-1], "BGR")

	scaled,unscaled=flatten(vec)
	screen.blit(pgimgv, (0,0))
	lx,ly=0,0
	for x,y in np.transpose(unscaled):
		#screen.set_at((int(x),int(y)),(255,0,0))
		pygame.draw.line(screen,(255,0,0),(lx,ly),(int(x),int(y)))
		lx,ly=int(x),int(y)


	vectosound(scaled).play()
	#print('len(flatten(vec)[0])',len(flatten(vec)[0]))
	pygame.display.flip()
