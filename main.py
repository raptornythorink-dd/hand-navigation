import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import pyautogui
import keyboard
import macmouse as mouse
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils #use drawing utility
handLmsStyle = mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1) #define landmark style
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1) #define connection style

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='hand_gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Gesture mapping
gesture_names = ["Fist", "Down", "Down-Left", "Left", "Up-Left", "Up", "Up-Right", "Right", "Down-Right", "Two Left", "Two Right", "Two Up", "Two Down", "Palm", "Hold", "Hand Left", "Hand Right", "Dog", "Reverse C", "OK", "Metal", "Line", "L", "Spock"]
click_gestures = {"Two Left", "Two Right"}
keyboard_gestures = {"Metal", "OK", "Reverse C", "Hand Left", "Hand Right", "Spock"}
special_gestures = {"Dog", "Line"}
width = 640
height = 360

# Screen size for mouse control
screen_width, screen_height = pyautogui.size()

# Queue to store last images for FPS calculation
max_len_queue = 10

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    # Take the first landmark as the reference point (0, 0)
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = np.array([[lm.x - base_x, lm.y - base_y] for lm in landmarks])
    return normalized.flatten()

# function calculate FPS
def calculate_fps(prev_time, prev_fps):
    current_time = time.time()
    fps = 0.9*prev_fps+ 0.1*(1 / (current_time - prev_time))
    return fps, current_time

#function find the bounding box
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

#function check if all last gestures are the same
def check_same_gestures(last_gestures):
    return all(gesture == last_gestures[0] for gesture in last_gestures)

def main():
    # Start capturing video from the camera
    cap = cv2.VideoCapture(0)
    enabled = False

    prev_time = time.time()
    prev_gesture_times = {gesture: time.time() for gesture in gesture_names}
    gesture_cooldowns = {gesture: (2 if gesture in special_gestures else 1.5 if gesture in click_gestures or gesture in keyboard_gestures else 0.1) for gesture in gesture_names}

    last_gestures = deque(maxlen=max_len_queue)  # Store last images for FPS calculation

    prev_fps=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))

        # Convert the frame to RGB as MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hands
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Normalize the landmarks
                normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)

                # Reshape and prepare input data
                input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(input_details[0]['shape'])

                # Set the input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get the output tensor
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Interpret the results
                predicted_class = np.argmax(output_data)
                if output_data[0][predicted_class] < 0.9:
                    continue
                gesture_name = gesture_names[predicted_class]
                print('Predicted gesture:', gesture_name)
                last_gestures.append(gesture_name)

                # Draw the hand landmarks on the frame
                mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, handLmsStyle,
                                    handConStyle)  # draw landmarks styles
                brect = calc_bounding_rect(frame, hand_landmarks)  # Calculate the bounding rectangle
                cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0),
                            1)  # Draw the bounding rectangle

                # Display the predicted gesture on the frame
                cv2.putText(frame, f'Gesture: {gesture_name}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)

                curr_time = time.time()
                if prev_gesture_times[gesture_name] + gesture_cooldowns[gesture_name] > curr_time:
                    continue

                prev_gesture_times[gesture_name] = curr_time

                move_speed = 0.01
                if gesture_name in ["Left", "Right", "Up", "Down", "Down-Left", "Down-Right", "Up-Left", "Up-Right"]:
                    lm = hand_landmarks.landmark

                    x_base, y_base = lm[5].x, lm[5].y
                    x_tip, y_tip = lm[8].x, lm[8].y

                    brect = calc_bounding_rect(frame, hand_landmarks)
                    hand_w = (brect[2] - brect[0]) / frame.shape[1]
                    hand_h = (brect[3] - brect[1]) / frame.shape[0]
                    hand_diag = np.sqrt(hand_w**2 + hand_h**2)

                    dist = np.sqrt((x_tip - x_base)**2 + (y_tip - y_base)**2)
                    norm_dist = dist / hand_diag
                    
                    move_speed = 0.05 * norm_dist

                if enabled and check_same_gestures(last_gestures):
                    match gesture_name:
                        case "Left":
                            mouse.move(-move_speed*screen_width, 0, False)  # Move mouse left
                        case "Right":
                            mouse.move(move_speed*screen_width, 0, False)  # Move mouse right
                        case "Up":
                            mouse.move(0, -move_speed*screen_height, False)  # Move mouse up
                        case "Down":
                            mouse.move(0, move_speed*screen_height, False)  # Move mouse down
                        case "Down-Left":
                            mouse.move(-move_speed*screen_width, move_speed*screen_height, False)
                        case "Down-Right":
                            mouse.move(move_speed*screen_width, move_speed*screen_height, False)
                        case "Up-Left":
                            mouse.move(-move_speed*screen_width, -move_speed*screen_height, False)
                        case "Up-Right":
                            mouse.move(move_speed*screen_width, -move_speed*screen_height, False)
                        case "Two Up":
                            mouse.wheel(-1)
                        case "Two Down":
                            mouse.wheel(1)
                        case "Two Left":
                            mouse.click()
                        case "Two Right":
                            mouse.right_click()
                        case "OK":
                            keyboard.send('enter')
                        case "Reverse C":
                            keyboard.send('escape')
                        case "Metal":
                            keyboard.send(['cmd', 75])
                        case "Left Hand":
                            keyboard.send('left')
                        case "Right Hand":
                            keyboard.send('right')
                        case _:
                            pass

                if gesture_name == "Dog":
                    enabled = not enabled
                    clockwise = 1 if enabled else -1
                    mouse.move(screen_width * 0.1 * clockwise, screen_height * 0.1 * clockwise, False, 0.1)
                    mouse.move(0, - screen_height * 0.2 * clockwise, False, 0.05)
                    mouse.move(- screen_width * 0.1 * clockwise, screen_height * 0.1 * clockwise, False, 0.1)
                    mouse.move(- screen_width * 0.1 * clockwise, - screen_height * 0.1 * clockwise, False, 0.1)
                    mouse.move(0, screen_height * 0.2 * clockwise, False, 0.05)
                    mouse.move(screen_width * 0.1 * clockwise, - screen_height * 0.1 * clockwise, False, 0.1)


        fps, prev_time = calculate_fps(prev_time, prev_fps)  # Calculate and display FPS
        prev_fps = fps
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Display the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
