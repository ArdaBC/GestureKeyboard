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