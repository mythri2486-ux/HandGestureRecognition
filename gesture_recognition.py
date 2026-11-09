import cv2
import mediapipe as mp
import math

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


# Calculate distance using pixel coordinates
def distance(a, b, width, height):
    ax, ay = a.x * width, a.y * height
    bx, by = b.x * width, b.y * height
    return math.hypot(ax - bx, ay - by)


# Check whether a normal finger is extended
def finger_extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y - 0.01


# Check whether thumb is extended
def thumb_extended(lm, width, height):
    tip_distance = distance(lm[4], lm[5], width, height)
    joint_distance = distance(lm[3], lm[5], width, height)

    return tip_distance > joint_distance * 1.15


# Recognize gestures
def recognize_gesture(lm, width, height):

    thumb = thumb_extended(lm, width, height)

    index = finger_extended(lm, 8, 6)
    middle = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pinky = finger_extended(lm, 20, 18)

    fingers = [thumb, index, middle, ring, pinky]
    count = sum(fingers)

    # Palm size for scale-independent OK detection
    palm_size = distance(lm[0], lm[9], width, height)

    thumb_index_distance = distance(lm[4], lm[8], width, height)

    # OK gesture
    if (
        palm_size > 0 and
        thumb_index_distance < palm_size * 0.35 and
        middle and ring and pinky
    ):
        gesture = "OK"

    # Open Palm
    elif count == 5:
        gesture = "Open Palm"

    # Fist
    elif count == 0:
        gesture = "Fist"

    # Peace
    elif fingers == [False, True, True, False, False]:
        gesture = "Peace"

    # Rock
    elif (
        index and pinky and
        not middle and not ring and
        not thumb
    ):
        gesture = "Rock"

    # Call Me
    elif (
        thumb and pinky and
        not index and not middle and not ring
    ):
        gesture = "Call Me"

    # Thumbs Up / Down
    elif thumb and not index and not middle and not ring and not pinky:

        if lm[4].y < lm[2].y - 0.03:
            gesture = "Thumbs Up"

        elif lm[4].y > lm[2].y + 0.03:
            gesture = "Thumbs Down"

        else:
            gesture = "Unknown"

    else:
        gesture = "Unknown"

    return count, gesture


# Main camera loop
while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Camera not detected!")
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    total = 0
    hand_count = 0

    if results.multi_hand_landmarks:

        hand_count = len(results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            count, gesture = recognize_gesture(lm, width, height)

            total += count

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Display gesture name
            x = int(lm[0].x * width)
            y = int(lm[0].y * height)

            cv2.putText(
                frame,
                gesture,
                (max(10, x - 50), max(30, y - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
            )

    # Display information
    cv2.putText(
        frame,
        "HANDS: " + str(hand_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "TOTAL FINGERS: " + str(total) + "/10",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Hand Gesture Recognition System", frame)

    # Press Q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()
q
