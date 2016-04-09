import cv2
import sys
import os

#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('/home/halwa/miniconda3/pkgs/opencv3-3.0.0-nppy27_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

path = './raw_images'
image_list = [image for image in os.listdir(path) if image.endswith(".jpg")]
i=0
for image in image_list:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        #minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        img_crop = img[y : y+h, x : x+w] 
        img_crop_name = str(i) + '.jpg'
        i += 1
        cv2.imwrite(img_crop_name, img_crop)
