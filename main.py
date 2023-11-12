import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import argparse
import keyboard

def quit():
    global running
    running = False

# Parse command-line arguments
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_camera', action='store_true')
    parser.add_argument('--show_hands', action='store_true')
    parser.add_argument('--sensitivity', type=float, default=2.0)
    parser.add_argument('--smoothness', type=int, default=2)
    parser.add_argument('--resolution', type=str, default='640x480')
    args = parser.parse_args()
    keyboard.add_hotkey('ctrl + q', quit)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    # Get the screen size
    screen_width, screen_height = pyautogui.size()

    # Set the sensitivity factor (adjust this value to increase or decrease sensitivity)
    sensitivity = args.sensitivity

    # Open the video stream
    cap = cv2.VideoCapture(0)

    # Reduce the resolution
    res1, res2 = tuple(map(int, args.resolution.split('x')))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res2)

    # Get the video stream size
    ret, frame = cap.read()
    video_width = frame.shape[1]
    video_height = frame.shape[0]

    # Define the boundaries of the virtual trackpad (center of the video stream)
    trackpad_width = int(video_width / sensitivity)
    trackpad_height = int(video_height / sensitivity)
    trackpad_top_left = (int((video_width - trackpad_width) / 2), int((video_height - trackpad_height) / 2))

    # Initialize smoothing factor
    smoothing_factor = args.smoothness  # Increase this value for more smoothing
    x_mov_avg = [0]*smoothing_factor
    y_mov_avg = [0]*smoothing_factor

    # Initialize mouse state
    mouse_down = False
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for better visualization
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (MediaPipe uses RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        # Draw the hand annotations on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if args.show_hands:
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Compute the center of the hand
                x_center = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x + hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x) / 2
                y_center = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y + hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) / 2

                # Map the hand position from the video coordinates to the trackpad coordinates
                x_hand_trackpad = (x_center * video_width - trackpad_top_left[0])
                y_hand_trackpad = (y_center * video_height - trackpad_top_left[1])

                # Check if the center of the hand is inside the virtual trackpad
                if 0 < x_hand_trackpad < trackpad_width and 0 < y_hand_trackpad < trackpad_height:
                    # Map the hand position from the trackpad coordinates to the screen coordinates
                    x_hand_screen = (x_hand_trackpad / trackpad_width) * screen_width
                    y_hand_screen = (y_hand_trackpad / trackpad_height) * screen_height

                    # Update moving averages
                    x_mov_avg.pop(0)
                    y_mov_avg.pop(0)
                    x_mov_avg.append(x_hand_screen)
                    y_mov_avg.append(y_hand_screen)

                    # Move the mouse cursor to correspond to the mapped hand position
                    pyautogui.moveTo(sum(x_mov_avg)/smoothing_factor, sum(y_mov_avg)/smoothing_factor)

                # Get the coordinates of the index fingertip and thumb tip
                index_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
                thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])

                # Compute the distance between the index fingertip and thumb tip
                distance = np.linalg.norm(index_finger_tip - thumb_tip)

                # If the distance is less than a threshold, consider the hand to be closed and click the mouse
                if distance < 0.1:
                    if not mouse_down:
                        pyautogui.mouseDown()
                        mouse_down = True
                else:
                    if mouse_down:
                        pyautogui.mouseUp()
                        mouse_down = False

        # Draw a rectangle on the screen that represents the area where the mouse cursor can move
        cv2.rectangle(frame, trackpad_top_left, (trackpad_top_left[0] + trackpad_width, trackpad_top_left[1] + trackpad_height), (0, 255, 0), 2)

        # Display the frame
        if args.show_camera or args.show_hands:
            cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
