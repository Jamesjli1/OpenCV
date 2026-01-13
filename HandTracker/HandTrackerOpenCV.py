import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
pTime = 0  # fps calculation

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)  # Thumb is open
    else:
        fingers.append(0)  # Thumb is closed

    # Other four fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips[id]].y < hand_landmarks.landmark[tips[id] - 2].y:
            fingers.append(1)  # Finger is open
        else:
            fingers.append(0)  # Finger is closed

    return fingers

def fingers_side(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingersq = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].y < hand_landmarks.landmark[tips[0] - 1].y:
        fingersq.append(1)  # Thumb is up
    else:
        fingersq.append(0)  # Thumb is down

    # Other four fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips[id]].x > hand_landmarks.landmark[tips[id] - 2].x:
            fingersq.append(1)  # Finger is to the right
        else:
            fingersq.append(0)  # Finger is to the left

    return fingersq

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if not success:
        break

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Get left/right hand label
            hand_name = hand_label.classification[0].label  # 'Left' or 'Right'

            finger_status = fingers_up(hand_landmarks)
            finger_status_side = fingers_side(hand_landmarks)
            gesture = ""
            
            if finger_status == [0,0,0,0,0]:
                gesture = "Fist "
            elif finger_status == [0,1,1,0,0]:
                gesture = "Peace "
            elif finger_status == [0,1,0,0,1]:
                gesture = "spideman "
            elif finger_status == [1,1,0,0,1]:
                gesture = "rock"
            elif finger_status == [1,0,1,0,0]:
                gesture = "middle finger"
            
            if finger_status_side == [1,1,0,0,0]:
                gesture = "index pointing right"
            elif finger_status_side == [1,0,1,1,1]:
                gesture = "index pointing left"
                    
            print(gesture)


            # Draw hand and show text
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Find bounding box for text placement
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)

            cv2.putText(frame, f"{hand_name}: {gesture}", (x_min - 20, y_min - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
