import macmouse as mouse
import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

HEIGHT, WIDTH = pyautogui.size()

def angle_between(a, b, c):
    ab = b - a
    cb = b - c
    ab = ab / np.linalg.norm(ab)
    cb = cb / np.linalg.norm(cb)
    dot = np.dot(ab, cb)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def is_finger_straight(hand_landmarks, min_index, w, h, proportion_threshold=0.5, angle_threshold=2.7):
    """
    Checks if the finger (starting at min_index) is pointed enough:
    - The distance between base and tip is a significant proportion of the hand size in that direction.
    - The finger is straight enough (angle check).
    """
    # Get base and tip coordinates
    base = hand_landmarks.landmark[min_index]
    tip = hand_landmarks.landmark[min_index + 3]
    base_xy = np.array([base.x * w, base.y * h])
    tip_xy = np.array([tip.x * w, tip.y * h])
    delta = tip_xy - base_xy

    # Get hand bounding box size
    all_points = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
    bbox_min = np.min(all_points, axis=0)
    bbox_max = np.max(all_points, axis=0)
    hand_size = bbox_max - bbox_min

    # Proportion check
    proportion_x = abs(delta[0]) / (hand_size[0] + 1e-6)
    proportion_y = abs(delta[1]) / (hand_size[1] + 1e-6)
    proportion_ok = proportion_x > proportion_threshold or proportion_y > proportion_threshold

    # Angle check (as before)
    index_positions = list(map(lambda l: np.array((l.x * w, l.y * h)), hand_landmarks.landmark[min_index:min_index + 4]))
    angle_ok = angle_between(*index_positions[:-1]) > angle_threshold and angle_between(*index_positions[1:]) > 0.5

    return proportion_ok and angle_ok


def main():
    cap = cv2.VideoCapture(0)
    last_left_click = time.time()
    last_right_click = time.time()
    cooldown = 1
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # 'Left' or 'Right'
                    if label == 'Left': # Actually right hand (because of flipping?)
                        color = (0, 0, 255)  # Red for right hand
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=color, thickness=2))
                        h, w, _ = image.shape
                        if is_finger_straight(hand_landmarks, 1, w, h, 0.5): # thumb
                            base, tip = hand_landmarks.landmark[1], hand_landmarks.landmark[4]
                            base_xy = np.array([base.x * w, base.y * h])
                            tip_xy = np.array([tip.x * w, tip.y * h])
                            direction = tip_xy - base_xy
                            norm = np.linalg.norm(direction)
                            if norm > 0:
                                if abs(direction[1]) > 0.7:
                                    if direction[1] > 0.7:
                                        if time.time() - last_right_click > cooldown:
                                            last_right_click = time.time()
                                            mouse.right_click()
                                            print("Right click")
                                    elif direction[1] < -0.7:
                                        if time.time() - last_left_click > cooldown:
                                            last_left_click = time.time()
                                            mouse.click()
                                            print("Left click")
                        if is_finger_straight(hand_landmarks, 5, w, h): 
                            base = hand_landmarks.landmark[5]
                            tip = hand_landmarks.landmark[8]
                            base_xy = np.array([base.x * w, base.y * h])
                            tip_xy = np.array([tip.x * w, tip.y * h])
                            direction = tip_xy - base_xy
                            norm = np.linalg.norm(direction)
                            if norm > 0:
                                direction = direction / norm
                                if abs(direction[1]) > 0.7:
                                    if direction[1] > 0.7:
                                        mouse.wheel(1)  # Up
                                    elif direction[1] < -0.7:
                                        mouse.wheel(-1)   # Down

                    else: # Left hand
                        color = (0, 255, 0)  # Green for left hand
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=color, thickness=2))
                        h, w, _ = image.shape
                        if is_finger_straight(hand_landmarks, 5, w, h, 0.2): # index
                            base, tip = hand_landmarks.landmark[5], hand_landmarks.landmark[8]
                            base_xy = np.array([base.x * w, base.y * h])
                            tip_xy = np.array([tip.x * w, tip.y * h])
                            direction = tip_xy - base_xy
                            norm = np.linalg.norm(direction)
                            if norm > 0:
                                direction = direction / norm
                                if abs(direction[0]) > 0.7:
                                    if direction[0] > 0.7:
                                        mouse.move(-WIDTH * 0.01, 0, False)
                                    elif direction[0] < -0.7:
                                        mouse.move(WIDTH * 0.01, 0, False)
                                elif abs(direction[1]) > 0.7:
                                    if direction[1] > 0.7:
                                        mouse.move(0, HEIGHT * 0.01, False)
                                    elif direction[1] < -0.7:
                                        mouse.move(0, -HEIGHT * 0.01, False)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))



            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()


if __name__ == "__main__":
    main()
