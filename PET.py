import cv2
import mediapipe as mp

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Finger tip landmark IDs (based on MediaPipe hand landmarks)
finger_tips_ids = [4, 8, 12, 16, 20]

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get label: Left or Right
            hand_label = hand_info.classification[0].label

            # Store landmark positions
            landmarks = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            # Count fingers
            fingers = []

            # Thumb (direction depends on hand)
            if hand_label == 'Right':
                fingers.append(landmarks[4][0] > landmarks[3][0])  # Right hand thumb
            else:
                fingers.append(landmarks[4][0] < landmarks[3][0])  # Left hand thumb

            # Other four fingers
            for tip_id in finger_tips_ids[1:]:
                fingers.append(landmarks[tip_id][1] < landmarks[tip_id - 2][1])

            # Add to total
            total_fingers += fingers.count(True)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display finger count
    cv2.putText(frame, f'Fingers: {total_fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 3)

    cv2.imshow("Hand & Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
