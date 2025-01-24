import numpy 
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import datetime
from collections import deque
from twilio.rest import Client 
import random
import heapq
import requests



def is_person_present(frame, thresh=1100):
    kernel = None
    global foog
    
    # Apply background subtraction
    fgmask = foog.apply(frame)

    # Get rid of the shadows
    ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Apply some morphological operations to make sure you have a good mask
    fgmask = cv2.dilate(fgmask, kernel, iterations=4)

    # Detect contours in the frame
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
    # Check if there was a contour and the area is somewhat higher than some threshold so we know it's a person and not noise
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > thresh:
            
        # Get the max contour
        cnt = max(contours, key=cv2.contourArea)

        # Draw a bounding box around the person and label it as person detected
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Person Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        
        return True, frame
        
        
    # Otherwise report there was no one present
    else:
        return False, frame


def send_message(body, info_dict):

    # Your Account SID from twilio.com/console
    account_sid = info_dict['account_sid']

    # Your Auth Token from twilio.com/console
    auth_token  = info_dict['auth_token']


    client = Client(account_sid, auth_token)

    message = client.messages.create(to=info_dict['your_num'], from_=info_dict['trial_num'], body=body)





def calculate_percentage_change(frame1, frame2):
    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between the two frames
    abs_diff = cv2.absdiff(gray_frame1, gray_frame2)

    # Calculate percentage change
    percentage_change = (np.count_nonzero(abs_diff) / abs_diff.size) * 100

    return percentage_change

def recogniseFromFace(frame): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if prediction[1] < 500:
            # print('deteccted')
            cv2.putText(frame, '%s' % (names[prediction[0]]), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0))
            return True ,names[prediction[0]]
        else:
            cv2.putText(frame, 'not recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            return False ,"not recognized"


def save_frame(frame_heap): 
    frame_index = 0
    while frame_heap:
        # Pop the frame with the maximum percentage change (negated)
        neg_change_percentage, frame , date = heapq.heappop(frame_heap)
        change_percentage = -neg_change_percentage
        file_path = 'DetectedPerson'
        # Save the frame to disk
        cv2.imwrite(f'{file_path}/{date}.jpg', frame)
        frame_index += 1
    print("saved frame succesfully")

print('no error in functions')




# Set Window normal so we can resize it
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# This is a test video
cap = cv2.VideoCapture('sample_video1.mp4')

# Read the video stream from the camera
# cap = cv2.VideoCapture('http://192.168.43.1:8080/video')

# Get width and height of the frame
width = int(cap.get(3))
height = int(cap.get(4))

# Read and store the credentials information in a dict
with open('credential.txt', 'r') as myfile:
  data = myfile.read()

info_dict = eval(data)

# Initialize the background Subtractor
foog = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=100, history=2000)

# Status is True when person is present and False when the person is not present.
status = False

# After the person disappears from view, wait at least 7 seconds before making the status False
patience = 0.099

# We don't consider an initial detection unless it's detected 15 times, this gets rid of false positives
detection_thresh = 5

# Initial time for calculating if patience time is up
initial_time = None



# We are creating a deque object of length detection_thresh and will store individual detection statuses here
de = deque([False] * 10, maxlen=10)

# Initialize an empty heap to store frames along with their negated percentage change
frame_heap = []


# Initialize these variables for calculating FPS
fps = 0 
frame_counter = 0
start_time = time.time()
counter = 0

ret,prevframe = cap.read()
initial_frame = prevframe
recognisedName = ""
sent  = False

while True:
    
    ret, frame = cap.read()
    if not ret:
        break 
    
            
    # This function will return a boolean variable telling if someone was present or not, it will also draw boxes if it 
    # finds someone
    detected, annotated_image = is_person_present(frame)  
    # print(recognisedName)
    if detected and len(recognisedName) == 0: 
        result = recogniseFromFace(frame)  
        if result is not None: 
            val,name = result 
            if val: 
                recognisedName = name 
            else: 
                recognisedName = ""
    # Register the current detection status on our deque object
    de.appendleft(detected)
    
    # If we have consecutively detected a person 15 times then we are sure that someone is present    
    # We also make this is the first time that this person has been detected so we only initialize the videowriter once
    if sum(de) == detection_thresh and not status:                       
            status = True
            entry_time = datetime.datetime.now().strftime("%A, %I-%M-%S %p %d %B %Y")
            # out = cv2.VideoWriter('outputs/{}.mp4'.format(entry_time), cv2.VideoWriter_fourcc(*'XVID'), 15.0, (width, height))

    # If status is True but the person is not in the current frame
    if status and not detected:
        
        # Restart the patience timer only if the person has not been detected for a few frames so we are sure it wasn't a 
        # False positive
        print('i reached deepest of the darkest point')
        if sum(de) > (detection_thresh/2): 
            if initial_time is None:
                print('initial time as been setted to',initial_time)
                initial_time = time.time()
                
            
            elif initial_time is not None:        
                print('now the initial time is not null beacuse current time is ',initial_time)
                print(time.time() - initial_time)
                # If the patience has run out and the person is still not detected then set the status to False
                # Also save the video by releasing the video writer and send a text message.
                if  time.time() - initial_time >= patience:
                    status = False
                    print("inside the loop")
                    exit_time = datetime.datetime.now().strftime("%A, %I:%M:%S %p %d %B %Y")
                    # out.release()
                    # print(out)
                    initial_time = None
                    de = deque([False] * len(de))
                    body = "Alert: \nA Person Entered the Room at {} \nLeft the room at {}".format(entry_time, exit_time) 
                    if len(recognisedName) > 0: 
                        body = "Alert: {} Has entered in your Room at {} \n Left the room at {}".format(recognisedName,entry_time,exit_time)
                    print('message has been sent')
                    send_message(body, info_dict)
                    frame_index = 0
                    cv2.destroyWindow("frame")
                    while frame_heap:
                        # Pop the frame with the maximum percentage change (negated)
                        neg_change_percentage, frame , date = heapq.heappop(frame_heap)
                        change_percentage = -neg_change_percentage
                        file_path = 'DetectedPerson'
                        
                        # Save the frame to disk
                        cv2.imwrite(f'{file_path}/{date}.jpg', frame)
                        frame_index += 1
                    print("saved frame succesfully")
                    continue
                    
                
    
    # If a significant amount of detections (more than half of detection_thresh) has occurred then we reset the Initial Time.
    elif status and sum(de) > (detection_thresh/2):
        change_percentage = calculate_percentage_change(prevframe, frame)
        change_initial_change = calculate_percentage_change(initial_frame,frame)
        # print(change_percentage)
        if change_percentage > 95 and sent == False: 
            sent = True
            body = "Alert: \n Some one has changed the camera configuration at:{}".format(datetime.datetime.now().strftime("%A, %I-%M-%S %p %d %B %Y")) 
            send_message(body,info_dict)
    
        if change_initial_change > 60: 
            heapq.heappush(frame_heap, (-change_percentage, frame,datetime.datetime.now().strftime("%A, %I-%M-%S %p %d %B %Y")))
        prevframe = frame
        initial_time = None
    
    # Get the current time in the required format
    current_time = datetime.datetime.now().strftime("%A, %I:%M:%S %p %d %B %Y")

    # Display the FPS
    cv2.putText(annotated_image, 'FPS: {:.2f}'.format(fps), (510, 450), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 40, 155), 2)
    
    # Display Time
    cv2.putText(annotated_image, current_time, (310, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)    
    
    # Display the Room Status
    cv2.putText(annotated_image, 'Room Occupied: {}'.format(str(status)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (200, 10, 150), 2)

    # Show the patience Value
    if initial_time is None:
        text = 'Patience: {}'.format(patience)
    else: 
        text = 'Patience: {:.2f}'.format(max(0, patience - (time.time() - initial_time)))
        
    cv2.putText(annotated_image, text, (10, 450), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 40, 155), 2)   

 
    # Show the Frame
    cv2.imshow('frame', frame)
    
    # Calculate the Average FPS
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))
    
    
    # Exit if q is pressed.
    if cv2.waitKey(30) == ord('q'):
        break


# Release Capture and destroy windows
cap.release()
cv2.destroyAllWindows()


