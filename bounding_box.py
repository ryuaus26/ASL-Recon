import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert the BGR image to RGB and process it with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Calculate average position of hand landmarks
            average_position = np.array([np.mean([landmark.x for landmark in hand_landmarks.landmark]), np.mean([landmark.y for landmark in hand_landmarks.landmark])])
            # Scale average position to actual image dimensions
            average_position = average_position * np.array([frame.shape[1], frame.shape[0]])
            # Calculate spread of hand landmarks
            min_x = min([landmark.x for landmark in hand_landmarks.landmark])
            max_x = max([landmark.x for landmark in hand_landmarks.landmark])
            min_y = min([landmark.y for landmark in hand_landmarks.landmark])
            max_y = max([landmark.y for landmark in hand_landmarks.landmark])
            # Draw bounding box around average position of hand landmarks
            spread = max(max_x - min_x, max_y - min_y)
            x = int(average_position[0] - spread / 2)
            y = int(average_position[1] - spread / 2)
            w = int(spread)
            h = int(spread)
            # Set minimum size of bounding box
            min_size = 100
            # Adjust width and height of bounding box if they are smaller than minimum size
            w = max(w, min_size)
            h = max(h, min_size)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

