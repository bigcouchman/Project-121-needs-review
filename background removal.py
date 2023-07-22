# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('mount everest.jpg')

# resizing the mountain image as 640 X 480
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file=cv2.VideoWriter(mountain,fourcc, 20.0,(640,480))

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([100,100,100])
        upper_bound = np.array([110,110,110])
        mask_one = cv2.inRange(frame_rgb,lower_bound,upper_bound)
        lower_bound = np.array([245,245,245])
        upper_bound = np.array([255,255,255])
        # thresholding image
        mask_two=cv2.inRange(frame_rgb,lower_bound,upper_bound)
        # inverting the mask
        mask=mask_one+mask_two
        cv2.imshow('mask 1',mask)
        mask_one=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask_one=cv2.morphologyEx(mask_one,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
        cv2.waitKey(1)
        # bitwise and operation to extract foreground / person
        mask_two=cv2.bitwise_not(mask_one)
        res_one=cv2.bitwise_and(frame,frame,mask=mask_two)
        res_two=cv2.bitwise_and(output_file,output_file,mask=mask_one)
        # final image
        final_output = cv2.addWeighted(res_one,1,res_two,1,0)
        output_file.write(frame)
        # show it
        cv2.imshow('frame' , final_output)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
output_file.release()
cv2.destroyAllWindows()
