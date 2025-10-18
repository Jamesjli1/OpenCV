
import cv2

#open webcam
cap = cv2.VideoCapture(0)

# Define colour range of red
#in a HSV colour space (color, intensity, brightness)
#this is limits of acceptable HSV values for red
lower_red = (0, 120, 70)
upper_red = (10, 255, 255)

"""
mask highlights where red is and makes it white
clean the mast to remove noise
find contours (outlines) of red objects
draw a box around the largest red object
compare center of box to center of frame (error)
"""


while True:
    #reads one frame, returns if value was grabbed and the frame itself(a numpy array)
    ret, frame = cap.read()

    #safety check to make sure frame was grabbed
    if not ret:
        print("Failed to grab frame")
        break

    # Get frame dimensions
    #shape returns (height, width, channels)
    #we only need height and width for center calculation
    frame_height, frame_width = frame.shape[:2] 

    # Calculating center of frame
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    # Convert frame to HSV (easier to isolate colours)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Creating mask for red colour
    #binary mask that makes red color range white(255)
    #everything else black(0)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Cleaning mask
    #mask is cleaned to remove noise(small dots of white)
    #small brush that slides across the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #erosion removes small white dots that are isolated
    #erosion would make tiny dots disappear
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #dilation makes white areas larger again
    #the other white areas (not dots) will grow back
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Blue crosshair in center
    #makes radius 5 circle at center in blue (255,0,0)
    #-1 means filled circle
    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)

    # Finding contours in the mask
    #boundares of white(red) areas in the mask
    #trace the borders of every red blob
    #each contour is a numpy array of (x,y) coordinates
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Finding largest contour by area
        #selecting the largest red object (target)
        largest_contour = max(contours, key=cv2.contourArea)

        # Bounding rectangle around it
        #finds smallest rectangle to fit around contour
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
        #draw rectangle in green (0,255,0)
        cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), )

        # Target center
        #finds center of rectangle
        center_x = x_rect + w_rect // 2
        center_y = y_rect + h_rect // 2

        # Coordinate text onscreen
        #finds center coordinates of target
        text = f"x: {center_x}, Y: {center_y}"
        #displays text in yellow (0,255,255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Error calculation from target center
        #finds how far target is from center of frame
        #used to move camera to center target(color)
        error_x = center_x - frame_center_x
        error_y = center_y - frame_center_y
        print("Error X:", error_x, "Error Y:", error_y)

    # Mask in the original frame
    #showing only the red parts of the frame
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Show frame
    cv2.imshow("Webcam", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources safely
cap.release()
cv2.destroyAllWindows()
