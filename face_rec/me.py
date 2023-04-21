import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture('crowd_video.mp4')

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Apply background subtraction to extract the foreground
    fgmask = fgbg.apply(frame)
    
    # Perform morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of the foreground objects
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours and filter out the small ones
    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue
        
        # Compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Draw the bounding box around the contour
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('Crowd Detection', frame)
    
    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
