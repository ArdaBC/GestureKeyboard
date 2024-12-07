import cv2
import mediapipe as mp
import math
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyautogui

# Initialize MediaPipe Hand solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
caps_lock_enabled = True  # Start with uppercase letters

# Get actual screen size
screen_width, screen_height = pyautogui.size()


# Define groups of letters with lowercase English characters and symbols
letter_groups = [
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
    ['j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r'],
    ['s', 't', 'u', 'v', 'w', 'x', 'y', 'z', ','],
    ['.', '?', '!', ':', ';', '-', '_', '(', ')']
]

# Define special buttons to be placed in the center
special_buttons = ['Enter', 'Del', 'Clear', 'Close', 'Space']

current_group_index = 0  # Start with the first group

# Initialize typed text
typed_text = ''
max_lines = 5  # Maximum lines to display

# Load a font that supports Turkish characters
# Update the path to a font that exists on your system
font_path = "times.ttf"  # Update to a valid font path on your system
font_large = ImageFont.truetype(font_path, 40)
font_medium = ImageFont.truetype(font_path, 30)
font_small = ImageFont.truetype(font_path, 24)

# Pre-render button labels to images to avoid rendering every frame
def create_text_image(text, font, size):
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # Use font.getbbox() to get the bounding box of the text
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(text_position, text, font=font, fill=(0, 0, 0))
    return np.array(image)

# Create a cache for button images
button_cache = {}
def get_button_image(text, radius, is_hovered, is_special=False):
    key = (text, radius, is_hovered, is_special)
    if key in button_cache:
        return button_cache[key]
    # Create button image
    size = (radius * 2, radius * 2)
    button_image = np.ones((size[1], size[0], 4), dtype=np.uint8) * 200
    color = (180, 220, 255, 255) if is_hovered else (200, 200, 200, 255) if not is_special else (220, 220, 220, 255)
    cv2.circle(button_image, (radius, radius), radius, color, -1)
    cv2.circle(button_image, (radius, radius), radius, (0, 0, 0, 255), 2)
    # Add text
    font = font_large if not is_special else font_medium
    text_image = create_text_image(text, font, size)
    alpha = text_image[:, :, 3] / 255.0
    for c in range(3):
        button_image[:, :, c] = button_image[:, :, c] * (1 - alpha) + text_image[:, :, c] * alpha
    button_cache[key] = button_image
    return button_image

def detect_gesture(hand_landmarks):
    """Detect specific gestures based on hand landmarks."""
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

        # Helper to check if a finger is extended
        def is_finger_extended(tip, pip):
            return tip.y < pip.y  # y is inverted in image coordinates

        # Determine finger states
        thumb_extended = thumb_tip.x > thumb_ip.x  # Thumb extended horizontally
        index_extended = is_finger_extended(index_tip, index_pip)
        middle_extended = is_finger_extended(middle_tip, middle_pip)
        ring_extended = is_finger_extended(ring_tip, ring_pip)
        pinky_extended = is_finger_extended(pinky_tip, pinky_pip)
                                            if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "Stop Sign Gesture"

        # Scroll Gesture: index and middle fingers extended, others not extended
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "Scroll Gesture"

        # Click Gesture: thumb and index finger close
        distance_thumb_index = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        if distance_thumb_index < 0.04:  # Adjust threshold as necessary
            return "Click Gesture"

        # Move Gesture: only index finger extended
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Move Gesture"

        return "Unknown Gesture"
    except Exception as e:
        print(f"Error in gesture detection: {e}")
        return "Unknown Gesture"


# Initialize webcam feed
cap = cv2.VideoCapture(0)
# Set the camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_x, prev_y = screen_width // 2, screen_height // 2
smoothing_factor = 0.2
last_click_time = 0
click_cooldown = 1  # Cooldown in seconds

scroll_active = False
scroll_start_y = None

running = True  # Flag to control the main loop

# Variables to manage fullscreen toggle
fullscreen = False

# Create the OpenCV window
cv2.namedWindow('Circular Hand Gesture Keyboard', cv2.WINDOW_NORMAL)

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    last_stop_sign_x = None  # Track the x-position of the hand for Stop Sign Gesture
    horizontal_scroll_threshold = 0.03  # Adjust threshold for normalized x-axis movement

    while running:
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape

        # Process hand landmarks
        results = hands.process(frame_rgb)
        gesture_name = "Unknown Gesture"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the camera feed
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                gesture_name = detect_gesture(hand_landmarks)

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                x_norm = index_tip.x
                y_norm = index_tip.y

                screen_x = int(x_norm * screen_width)
                screen_y = int(y_norm * screen_height)

                smooth_x = prev_x + (screen_x - prev_x) * smoothing_factor
                smooth_y = prev_y + (screen_y - prev_y) * smoothing_factor

                if gesture_name == "Stop Sign Gesture":
                    # Track horizontal movement for scrolling
                    current_stop_sign_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x  # Use wrist for x-coordinate

                    if last_stop_sign_x is not None:
                        dx = current_stop_sign_x - last_stop_sign_x
                        if dx > horizontal_scroll_threshold:  # Scroll Right
                            if caps_lock_enabled:
                                caps_lock_enabled = False
                                letter_groups = [[letter.lower() for letter in group] for group in letter_groups]
                                print("Caps Lock OFF: Letters are now lowercase")
                                print("Stop Sign: Scrolled Right")
                        elif dx < -horizontal_scroll_threshold:  # Scroll Left
                            if not caps_lock_enabled:   
                                caps_lock_enabled = True
                                letter_groups = [[letter.upper() for letter in group] for group in letter_groups]
                                print("Caps Lock ON: Letters are now uppercase")
                                print("Stop Sign: Scrolled Left")

                    last_stop_sign_x = current_stop_sign_x
                else:
                    last_stop_sign_x = None  # Reset when not using Stop Sign Gesture
                    
                if gesture_name == "Move Gesture":
                    prev_x, prev_y = smooth_x, smooth_y

                elif gesture_name == "Click Gesture" and (time.time() - last_click_time > click_cooldown):
                    last_click_time = time.time()
                    # Check if a button is clicked
                    for button in button_positions:
                        dist = math.hypot(smooth_x - button['x'], smooth_y - button['y'])
                        if dist <= button['radius']:
                            letter = button['letter']
                            if letter == 'Space':
                                typed_text += ' '
                            elif letter == 'Del':
                                typed_text = typed_text[:-1]
                            elif letter == 'Enter':
                                typed_text += '\n'
                            elif letter == 'Clear':
                                typed_text = ''
                            elif letter == 'Close':
                                running = False
                            else:
                                typed_text += letter
                            print(f"Button '{letter}' clicked!")
                            break

                elif gesture_name == "Scroll Gesture":
                    if not scroll_active:
                        scroll_active = True
                        scroll_start_y = smooth_y
                    else:
                        dy = smooth_y - scroll_start_y
                        if dy < -15:  # Scroll Up
                            if current_group_index > 0:
                                current_group_index -= 1
                                print("Scrolled Up")
                            scroll_start_y = smooth_y
                        elif dy > 15:  # Scroll Down
                            if current_group_index < len(letter_groups) - 1:
                                current_group_index += 1
                                print("Scrolled Down")
                            scroll_start_y = smooth_y
                else:
                    scroll_active = False
        else:
            scroll_active = False

        # Create a blank image for the interface
        interface = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 200

        # Draw circular keyboard on the interface
        center_x = interface.shape[1] // 2
        center_y = interface.shape[0] // 2 + 100  # Adjusted as needed
        radius = 300  # Radius of the circle where buttons are placed

        current_letters = letter_groups[current_group_index]
        cursor_pos = (int(prev_x), int(prev_y))

        button_positions = []

        # Draw letters around the circle
        n = len(current_letters)
        angle_step = 2 * math.pi / n
        button_radius = 60  # Radius of each button

        for i in range(n):
            angle = i * angle_step - math.pi / 2  # Start from the top
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))

            # Check if cursor is over the button
            dist_to_cursor = math.hypot(cursor_pos[0] - x, cursor_pos[1] - y)
            is_hovered = dist_to_cursor <= button_radius

            # Get button image
            button_image = get_button_image(current_letters[i], button_radius, is_hovered)
            # Overlay button image on interface
            y1, y2 = y - button_radius, y + button_radius
            x1, x2 = x - button_radius, x + button_radius

            # Ensure the coordinates are within the interface boundaries
            y1_clipped = max(0, y1)
            y2_clipped = min(interface.shape[0], y2)
            x1_clipped = max(0, x1)
            x2_clipped = min(interface.shape[1], x2)

            button_y1 = int(y1_clipped - y1)
            button_y2 = int(y2_clipped - y1)
            button_x1 = int(x1_clipped - x1)
            button_x2 = int(x2_clipped - x1)

            if y1_clipped < y2_clipped and x1_clipped < x2_clipped:
                try:
                    alpha_s = button_image[button_y1:button_y2, button_x1:button_x2, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(3):
                        interface[int(y1_clipped):int(y2_clipped), int(x1_clipped):int(x2_clipped), c] = (
                            alpha_s * button_image[button_y1:button_y2, button_x1:button_x2, c] +
                            alpha_l * interface[int(y1_clipped):int(y2_clipped), int(x1_clipped):int(x2_clipped), c]
                        )
                except Exception as e:
                    print(f"Error overlaying button image: {e}")

            # Store button position for hit testing
            button_positions.append({
                'x': x,
                'y': y,
                'radius': button_radius,
                'letter': current_letters[i]
            })