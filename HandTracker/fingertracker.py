# Tracks the index finger of the left hand and calculates its position error relative to the center of the screen.

import cv2
import mediapipe as mp
import time

TARGET_HAND_LABEL = "Left" 
SMOOTHING_FACTOR = 0.2      # 0.0 to 1.0 (Higher = faster response, Lower = smoother)

# MediaPipe Hands Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hands object with specified parameters
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Initialize previous error values for smoothing
prev_error_x = 0.0
prev_error_y = 0.0
target_detected = False

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert color space for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand Detection
    result = hands.process(rgb_frame)
    
    # Get frame dimensions
    h, w, _ = frame.shape
    center_x = w // 2
    center_y = h // 2

    # Vertical Center Line (for X error reference)
    cv2.line(frame, (center_x, 0), (center_x, h), (100, 100, 100), 1)
    # Horizontal Center Line (for Y error reference)
    cv2.line(frame, (0, center_y), (w, center_y), (100, 100, 100), 1)

    target_detected = False
    
    # If hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
            
            # Check for correct hand
            if hand_label.classification[0].label == TARGET_HAND_LABEL:
                target_detected = True
                
                # Extract Index Finger Tip (Landmark 8)
                index_tip = hand_landmarks.landmark[8]
                
                # Convert to pixel coordinates
                px, py = int(index_tip.x * w), int(index_tip.y * h)

                # X Error: (Left = -1, Right = 1)
                raw_error_x = (index_tip.x - 0.5) * 2
                # Y Error: (Bottom  = -1, Top = 1)
                raw_error_y = (0.5 - index_tip.y) * 2

                # Smooth the Output (Exponential Moving Average) 
                curr_error_x = (SMOOTHING_FACTOR * raw_error_x) + ((1 - SMOOTHING_FACTOR) * prev_error_x)
                curr_error_y = (SMOOTHING_FACTOR * raw_error_y) + ((1 - SMOOTHING_FACTOR) * prev_error_y)
                
                # Update previous values for next loop
                prev_error_x = curr_error_x
                prev_error_y = curr_error_y

                # Draw Hand Landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw Circle on Index Tip
                cv2.circle(frame, (px, py), 8, (0, 255, 0), cv2.FILLED)
                
                # Show Error Lines
                color_x = (0, 0, 255) if abs(curr_error_x) > 0.1 else (0, 255, 0) # color
                cv2.line(frame, (center_x, py), (px, py), color_x, 2)             # draw line
                color_y = (0, 0, 255) if abs(curr_error_y) > 0.1 else (0, 255, 0) 
                cv2.line(frame, (px, center_y), (px, py), color_y, 2)

                # Display error values
                cv2.putText(frame, f"X Err: {curr_error_x:.2f}", (px + 10, py - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_x, 2)
                cv2.putText(frame, f"Y Err: {curr_error_y:.2f}", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_y, 2)

    # UI Status
    status_text = "Target: LOCKED" if target_detected else "Target: SEARCHING"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Finger Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()