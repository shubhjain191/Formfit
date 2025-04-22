import cv2
import PoseModule2 as pm
import numpy as np
import streamlit as st
from AiTrainer_utils import *
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import random
import pandas as pd
from collections import defaultdict
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

relevant_landmarks_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    return np.abs(a[1] - b[1])

# Feedback message categories
POSITIVE_MESSAGES = {
    "push_up_arm": ["Perfect arm bend—textbook form!", "Great elbow depth!", "Arms are spot on!", "Smooth arm movement!", "Nailed the push-up angle!"],
    "push_up_back": ["Back's like a steel rod!", "Core locked in tight!", "Perfect plank alignment!", "Rock-solid posture!", "Back's a straight line!"],
    "squat_legs": ["Legs are powerhouse strong!", "Knee bend is perfect!", "Solid squat depth!", "Great leg control!", "Thighs are dialed in!"],
    "squat_back": ["Back's upright and steady!", "Posture's a 10!", "Chest up, back strong!", "Spine's perfectly aligned!", "Great back stability!"],
    "bicep_curl": ["Biceps are popping!", "Smooth curl, full range!", "Perfect arm isolation!", "Controlled and strong!", "Curls are flawless!"],
    "shoulder_press": ["Shoulders are crushing it!", "Full press, full power!", "Great overhead form!", "Arms locked out perfectly!", "Press is on point!"]
}

NEGATIVE_MESSAGES = {
    "push_up_arm": ["Bend elbows more for depth.", "Lower your chest closer.", "Arms need more range.", "Don't lock out early.", "Push deeper!"],
    "push_up_back": ["Straighten that back!", "Hips are sagging—lift up.", "Avoid dipping your core.", "Back's bending—tighten up!", "Keep it flat!"],
    "squat_legs": ["Squat lower—hit parallel!", "Knees too far forward.", "Hips need to drop more.", "Legs not deep enough.", "Push knees out!"],
    "squat_back": ["Keep back straight—don't lean!", "Chest up, don't slump!", "Back's tilting—fix it!", "Don't round your spine!", "Stay upright!"],
    "bicep_curl": ["Elbows drifting—keep them in!", "Don't swing—control it!", "Curl higher for range.", "Too fast—slow it down!", "Arms need more lift!"],
    "shoulder_press": ["Extend arms fully up!", "Don't lean back—stay tall!", "Lower arms more on reset.", "Press straighter!", "Arms not locked out!"]
}

IMPROVEMENT_TIPS = {
    "push_up_arm": ["Aim for a 90° elbow bend.", "Lower slow and controlled.", "Keep elbows close to body.", "Pause at the bottom."],
    "push_up_back": ["Engage core for a flat back.", "Imagine a plank hold.", "Squeeze glutes to lift hips.", "Focus on shoulder-hip alignment."],
    "push_up_symmetry": ["Even out arm angles.", "Push with both sides equally.", "Check your mirror image.", "Balance your effort."],
    "squat_legs": ["Push knees out slightly.", "Drive up through heels.", "Sink hips to knee level.", "Control the descent."],
    "squat_back": ["Pull shoulders back tight.", "Look forward, not down.", "Brace your core hard.", "Keep chest lifted."],
    "squat_symmetry": ["Match knee bends.", "Shift weight evenly.", "Check leg alignment.", "Stay centered."],
    "bicep_curl": ["Lock elbows by your sides.", "Lower weight slowly.", "Full curl to chest.", "Avoid momentum swings."],
    "bicep_symmetry": ["Sync both arms' motion.", "Lift weights evenly.", "Focus on weaker side.", "Mirror your curls."],
    "shoulder_press": ["Press straight overhead.", "Keep back flat, no arch.", "Lower to chin level.", "Engage shoulders fully."],
    "shoulder_symmetry": ["Even out arm extension.", "Press both sides together.", "Match your arm heights.", "Balance the load."]
}

MOTIVATIONAL_MESSAGES = [
    "You're unstoppable—keep it up!", 
    "Powering through like a champ!", 
    "Every rep's building strength!", 
    "You've got this—pure grit!", 
    "Beast mode activated!"
]

def count_repetition_push_up(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    right_shoulder = landmark_list[12][1:]
    left_shoulder = landmark_list[11][1:]
    
    # Visualize angles for better feedback
    exercise_instance.visualize_angle(img, right_arm_angle, right_shoulder)
    exercise_instance.visualize_angle(img, left_arm_angle, left_shoulder)
    
    # Use average angle for more robust detection
    avg_arm_angle = (right_arm_angle + left_arm_angle) / 2
    
    # More robust thresholds with hysteresis
    DOWN_THRESHOLD = 240
    UP_THRESHOLD = 250
    
    # State machine for push-up counting
    if avg_arm_angle < DOWN_THRESHOLD:
        if stage != "down":
            print(f"Push-up DOWN detected: {avg_arm_angle}")
            stage = "down"
    
    if avg_arm_angle > UP_THRESHOLD and stage == "down":
        stage = "up"
        counter += 1
        current_time = time.time()
        exercise_instance.log_rep_data('push_up', current_time, 
                                      {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
        print(f"Push-up rep {counter} logged at {current_time}")
        exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
        exercise_instance.last_activity_time = current_time
    
    return stage, counter

def count_repetition_squat(detector, img, landmark_list, stage, counter, exercise_instance):
    right_leg_angle = detector.find_angle(img, 24, 26, 28)
    left_leg_angle = detector.find_angle(img, 23, 25, 27)
    right_leg = landmark_list[26][1:]
    exercise_instance.visualize_angle(img, right_leg_angle, right_leg)
    if right_leg_angle > 150 and left_leg_angle < 200:
        stage = "down"
    if right_leg_angle < 150 and left_leg_angle > 180 and stage == "down":
        stage = "up"
        counter += 1
        current_time = time.time()
        exercise_instance.log_rep_data('squat', current_time, {'right_leg': right_leg_angle, 'left_leg': left_leg_angle})
        print(f"Squat rep {counter} logged at {current_time}")
        exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
        exercise_instance.last_activity_time = current_time
    return stage, counter

def count_repetition_bicep_curl(detector, img, landmark_list, stage_right, stage_left, counter, exercise_instance):
    # Calculate angles for both arms
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    
    # Visualize angles (only once, not from multiple functions)
    exercise_instance.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
    exercise_instance.visualize_angle(img, left_arm_angle, landmark_list[13][1:])
    
    # Get visual landmarks for both arms
    right_shoulder = landmark_list[12][1:]
    right_elbow = landmark_list[14][1:]
    right_wrist = landmark_list[16][1:]
    left_shoulder = landmark_list[11][1:]
    left_elbow = landmark_list[13][1:]
    left_wrist = landmark_list[15][1:]
    
    # Convert to pixel coordinates for visualization
    r_shoulder = tuple(np.multiply(right_shoulder, [640, 480]).astype(int))
    r_elbow = tuple(np.multiply(right_elbow, [640, 480]).astype(int))
    r_wrist = tuple(np.multiply(right_wrist, [640, 480]).astype(int))
    l_shoulder = tuple(np.multiply(left_shoulder, [640, 480]).astype(int))
    l_elbow = tuple(np.multiply(left_elbow, [640, 480]).astype(int))
    l_wrist = tuple(np.multiply(left_wrist, [640, 480]).astype(int))
    
    # Draw arm connections
    cv2.line(img, r_shoulder, r_elbow, (0, 255, 255), 2)
    cv2.line(img, r_elbow, r_wrist, (0, 255, 255), 2)
    cv2.line(img, l_shoulder, l_elbow, (0, 255, 255), 2)
    cv2.line(img, l_elbow, l_wrist, (0, 255, 255), 2)
    
    # Initialize stages if they're None
    if stage_right is None:
        stage_right = "none"
    if stage_left is None:
        stage_left = "none"
    
    # Check for DOWN position (arms extended)
    # Angle ranges for DOWN: typically 140-180 degrees at the elbow
    if right_arm_angle > 140 and right_arm_angle < 190:
        if stage_right != "down":
            print(f"Right arm DOWN: {right_arm_angle:.1f}")
        stage_right = "down"
    
    if left_arm_angle > 140 and left_arm_angle < 190:
        if stage_left != "down":
            print(f"Left arm DOWN: {left_arm_angle:.1f}")
        stage_left = "down"
    
    # Check for UP position (arms bent)
    # For UP position, either angle is very small (<70) or very large (>300) due to how angles are calculated
    if stage_right == "down" and stage_left == "down":
        # Both arms must be in UP position to count a rep
        if ((right_arm_angle < 70) or (right_arm_angle > 300)) and ((left_arm_angle < 70) or (left_arm_angle > 300)):
            stage_right = "up"
            stage_left = "up"
            counter += 1
            current_time = time.time()
            
            # Visual feedback for rep counting
            cv2.putText(img, "REP COUNTED!", (int(img.shape[1]/2)-100, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Log the repetition data
            exercise_instance.log_rep_data('bicep_curl', current_time, 
                                          {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
            
            print(f"Bicep curl rep {counter} counted! R:{right_arm_angle:.1f}° L:{left_arm_angle:.1f}°")
            
            # Reset feedback timing if available
            if hasattr(exercise_instance, 'last_feedback_time'):
                exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
            if hasattr(exercise_instance, 'last_activity_time'):
                exercise_instance.last_activity_time = current_time
    
    # Create a status display area
    status_bg = np.zeros((70, 200, 3), dtype=np.uint8)
    cv2.putText(status_bg, f"R: {stage_right.upper()}", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(status_bg, f"L: {stage_left.upper()}", (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Paste the status display onto the image
    x_offset = 10
    y_offset = 120
    img[y_offset:y_offset+status_bg.shape[0], x_offset:x_offset+status_bg.shape[1]] = status_bg
    
    return stage_right, stage_left, counter

def count_repetition_shoulder_press(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    right_elbow = landmark_list[14][1:]
    exercise_instance.visualize_angle(img, right_arm_angle, right_elbow)
    if right_arm_angle > 250 and left_arm_angle < 100:
        stage = "down"
    if right_arm_angle < 210 and left_arm_angle > 150 and stage == "down":
        stage = "up"
        counter += 1
        current_time = time.time()
        exercise_instance.log_rep_data('shoulder_press', current_time, {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
        print(f"Shoulder press rep {counter} logged at {current_time}")
        exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
        exercise_instance.last_activity_time = current_time
    return stage, counter

def generate_feedback(detector, img, landmark_list, exercise_type, stage, feedback_history, prev_angles, exercise_instance):
    form_feedback = []
    issues = defaultdict(int)
    current_angles = {}
    user_reps = exercise_instance.counters.get(exercise_type, 0)

    feedback_triggered = stage == "up"

    # Default positive feedback key mapping for each exercise
    default_positive_key = {
        'push_up': 'push_up_arm',
        'squat': 'squat_legs',
        'bicep_curl': 'bicep_curl',
        'shoulder_press': 'shoulder_press'
    }

    if exercise_type == 'push_up':
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        current_angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
        avg_arm_angle = (right_arm_angle + left_arm_angle) / 2
        symmetry_diff = abs(right_arm_angle - left_arm_angle)
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]]).mean(axis=0)
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]]).mean(axis=0)
        knees = np.array([landmark_list[25][1:], landmark_list[26][1:]]).mean(axis=0)
        back_angle = calculate_angle(shoulders, hips, knees)

        if stage == "down":
            if avg_arm_angle > 110:
                issues["arm_depth"] += 1
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["push_up_arm"]), False))
            elif avg_arm_angle < 70:
                issues["arm_too_low"] += 1
                form_feedback.append(("Don't go too low—ease up!", False))
            elif symmetry_diff > 20:
                issues["symmetry"] += 1
                form_feedback.append(("Arms uneven—sync them up!", False))
            elif back_angle < 160:
                issues["back"] += 1
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["push_up_back"]), False))
            else:
                form_feedback.append((random.choice(POSITIVE_MESSAGES["push_up_arm"]), True))

    elif exercise_type == 'squat':
        right_knee_angle = detector.find_angle(img, 24, 26, 28)
        left_knee_angle = detector.find_angle(img, 23, 25, 27)
        current_angles = {'right_leg': right_knee_angle, 'left_leg': left_knee_angle}
        avg_knee_angle = (right_knee_angle + left_knee_angle) / 2
        symmetry_diff = abs(right_knee_angle - left_knee_angle)
        shoulders = np.array([landmark_list[11][1:], landmark_list[12][1:]]).mean(axis=0)
        hips = np.array([landmark_list[23][1:], landmark_list[24][1:]]).mean(axis=0)
        knees = np.array([landmark_list[25][1:], landmark_list[26][1:]]).mean(axis=0)
        back_angle = calculate_angle(shoulders, hips, knees)

        if stage == "down":
            if avg_knee_angle > 105:
                issues["leg_depth"] += 1
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["squat_legs"]), False))
            elif avg_knee_angle < 65:
                issues["leg_too_low"] += 1
                form_feedback.append(("Too deep—back off a bit!", False))
            elif symmetry_diff > 20:
                issues["symmetry"] += 1
                form_feedback.append(("Legs uneven—align them!", False))
            elif back_angle < 150:
                issues["back"] += 1
                form_feedback.append((random.choice(NEGATIVE_MESSAGES["squat_back"]), False))
            else:
                form_feedback.append((random.choice(POSITIVE_MESSAGES["squat_legs"]), True))

    elif exercise_type == 'bicep_curl':
        right_shoulder_x = landmark_list[12][1]
        right_elbow_x = landmark_list[14][1]
        left_shoulder_x = landmark_list[11][1]
        left_elbow_x = landmark_list[13][1]
        right_arm_angle = detector.find_angle(img,12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        current_angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
        symmetry_diff = abs(right_arm_angle - left_arm_angle)

        if abs(right_elbow_x - right_shoulder_x) > 0.15 or abs(left_elbow_x - left_shoulder_x) > 0.15:
            issues["elbow_position"] += 1
            form_feedback.append((random.choice(NEGATIVE_MESSAGES["bicep_curl"]), False))
        elif symmetry_diff > 25:
            issues["symmetry"] += 1
            form_feedback.append(("Arms out of sync—match them!", False))
        elif right_arm_angle > 180 or left_arm_angle > 180:
            form_feedback.append(("Lower fully—great effort!", True))
        elif right_arm_angle < 90 and left_arm_angle < 90:
            form_feedback.append((random.choice(POSITIVE_MESSAGES["bicep_curl"]), True))
        else:
            form_feedback.append(("Solid curl—keep it up!", True))

    elif exercise_type == 'shoulder_press':
        back_angle = detector.find_angle(img, 11, 23, 25)
        right_arm_angle = detector.find_angle(img, 12, 14, 16)
        left_arm_angle = detector.find_angle(img, 11, 13, 15)
        current_angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
        symmetry_diff = abs(right_arm_angle - left_arm_angle)

        if back_angle < 165:
            issues["back"] += 1
            form_feedback.append((random.choice(NEGATIVE_MESSAGES["shoulder_press"]), False))
        elif symmetry_diff > 20:
            issues["symmetry"] += 1
            form_feedback.append(("Arms uneven—press evenly!", False))
        elif stage == "up" and (abs(right_arm_angle - 180) > 30 or abs(left_arm_angle - 180) > 30):
            issues["arm_extension"] += 1
            form_feedback.append(("Extend fully—reach up!", False))
        else:
            form_feedback.append((random.choice(POSITIVE_MESSAGES["shoulder_press"]), True))

    if feedback_triggered:
        if form_feedback:
            form_feedback.append((random.choice(MOTIVATIONAL_MESSAGES), True))
        else:
            # Use the default positive key for the exercise type
            positive_key = default_positive_key.get(exercise_type, exercise_type)
            form_feedback = [(random.choice(POSITIVE_MESSAGES[positive_key]), True), (random.choice(MOTIVATIONAL_MESSAGES), True)]

    if issues and random.random() < 0.5:
        for issue in issues:
            feedback_history[exercise_type][issue] += issues[issue]
            if feedback_history[exercise_type][issue] > 5:
                tip_key = f"{exercise_type}_{issue.split('_')[0]}" if issue != "symmetry" else f"{exercise_type}_symmetry"
                form_feedback.append((random.choice(IMPROVEMENT_TIPS.get(tip_key, ["Work on your form!"])), False))
                break

    if len(form_feedback) > 2:
        form_feedback = form_feedback[:2]

    significant_change = False
    if prev_angles:
        for key in current_angles:
            if abs(current_angles[key] - prev_angles.get(key, current_angles[key])) > 20:
                significant_change = True
                break

    return form_feedback, current_angles, significant_change

class Exercise:
    def __init__(self):
        try:
            self.lstm_model = load_model('final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5')
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            self.lstm_model = None
        try:
            self.scaler = joblib.load('thesis_bidirectionallstm_scaler.pkl')
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
        try:
            self.label_encoder = joblib.load('thesis_bidirectionallstm_label_encoder.pkl')
            self.exercise_classes = self.label_encoder.classes_
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            self.label_encoder = None
            self.exercise_classes = []
        self.form_feedback = []
        self.fatigue_feedback = []
        self.stop_requested = False
        self.rep_timestamps = []
        self.angle_history = {exercise: [] for exercise in ['push_up', 'squat', 'bicep_curl', 'shoulder_press']}
        self.fatigue_level = 0.0
        self.fatigue_trend = 0.0
        self.fatigue_warnings = []
        self.prev_form_feedback = []
        self.prev_fatigue_score = 0.0
        self.feedback_history = {
            'push_up': defaultdict(int),
            'squat': defaultdict(int),
            'bicep_curl': defaultdict(int),
            'shoulder_press': defaultdict(int)
        }
        self.last_feedback_time = time.time()
        self.feedback_interval = 2
        self.prev_angles = {}
        self.counters = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.sets = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}  # New sets tracker
        self.baseline_speeds = {}
        self.last_activity_time = time.time()
        
        # New tracking variables for set detection
        self.set_pause_threshold = 30  # 30 seconds of inactivity defines a new set
        self.last_rep_time = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.current_set_reps = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.set_history = {
            'push_up': [],     # Store rep counts for each set: [5, 8, 6] means 3 sets with 5, 8, 6 reps
            'squat': [],
            'bicep_curl': [],
            'shoulder_press': []
        }
        self.set_start_times = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.set_end_times = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.is_resting = {'push_up': False, 'squat': False, 'bicep_curl': False, 'shoulder_press': False}
        self.rest_start_time = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.rest_durations = {'push_up': [], 'squat': [], 'bicep_curl': [], 'shoulder_press': []}  # Track rest periods
        
        # Enhanced fatigue tracking per set
        self.set_fatigue_metrics = {
            'push_up': [],
            'squat': [],
            'bicep_curl': [],
            'shoulder_press': []
        }  # List of dicts with metrics for each set
        
        # Enhanced ROM targets with ideal ranges
        self.rom_targets = {
            'push_up': {'min': 70, 'max': 110, 'ideal': 90},
            'squat': {'min': 65, 'max': 105, 'ideal': 90},
            'bicep_curl': {'min': 80, 'max': 180, 'ideal': 150},
            'shoulder_press': {'min': 150, 'max': 210, 'ideal': 180}
        }
        
        # Per-set baselines to improve fatigue tracking between sets
        self.set_baselines = {
            'push_up': {'speed': {}, 'rom': {}, 'variability': {}},
            'squat': {'speed': {}, 'rom': {}, 'variability': {}},
            'bicep_curl': {'speed': {}, 'rom': {}, 'variability': {}},
            'shoulder_press': {'speed': {}, 'rom': {}, 'variability': {}}
        }
        
        # Specific tracking for shoulder press exercise
        self.shoulder_press_metrics = {
            'speed_history': [],
            'rom_history': [],
            'form_history': []
        }
        
        # New variables for enhanced rest period visualization and set transition
        self.rest_timer_start = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.show_set_transition = {'push_up': False, 'squat': False, 'bicep_curl': False, 'shoulder_press': False}
        self.transition_message = {'push_up': "", 'squat': "", 'bicep_curl': "", 'shoulder_press': ""}
        self.transition_start_time = {'push_up': 0, 'squat': 0, 'bicep_curl': 0, 'shoulder_press': 0}
        self.transition_duration = 3.0  # Show transition message for 3 seconds

    def log_rep_data(self, exercise_type, timestamp, angles):
        """Log data for a single repetition"""
        self.rep_timestamps.append(timestamp)
        self.angle_history[exercise_type].append(angles)
        
        # Update rep count and tracking for set detection
        current_time = timestamp
        
        # If this is the very first rep for this exercise type, initialize tracking
        if self.set_start_times[exercise_type] == 0:
            self.set_start_times[exercise_type] = current_time
            self.current_set_reps[exercise_type] = 1
            self.counters[exercise_type] = 1
            self.last_rep_time[exercise_type] = current_time
            self.is_resting[exercise_type] = False
            return
            
        # Check if we were in a resting state
        if self.is_resting[exercise_type]:
            # Calculate how long the rest was
            rest_duration = current_time - self.rest_start_time[exercise_type]
            self.rest_durations[exercise_type].append(rest_duration)
            
            # If rest exceeded threshold, complete the previous set and start a new one
            if rest_duration >= self.set_pause_threshold:
                # Only record the set if it had at least 1 rep
                if self.current_set_reps[exercise_type] > 0:
                    # Save the completed set
                    self.set_history[exercise_type].append(self.current_set_reps[exercise_type])
                    self.set_end_times[exercise_type] = self.last_rep_time[exercise_type]
                    self.sets[exercise_type] += 1
                    
                    # Start a new set
                    self.current_set_reps[exercise_type] = 1
                    self.set_start_times[exercise_type] = current_time
                    
                    # Log the set completion
                    print(f"Set {self.sets[exercise_type]} of {exercise_type} completed with {self.set_history[exercise_type][-1]} reps.")
                    print(f"Rest period: {rest_duration:.1f} seconds. Starting new set.")
            else:
                # Rest was shorter than threshold, continue the current set
                self.current_set_reps[exercise_type] += 1
                
            # No longer resting
            self.is_resting[exercise_type] = False
        else:
            # Normal rep within a set
            self.current_set_reps[exercise_type] += 1
        
        # Update total rep count and last rep time
        self.counters[exercise_type] = sum(self.set_history[exercise_type]) + self.current_set_reps[exercise_type]
        self.last_rep_time[exercise_type] = current_time
        self.last_activity_time = current_time

    def detect_fatigue(self, exercise_type, window_size=5, min_reps=3, weight_factor=1.0):
        """
        Enhanced fatigue detection with accurate tracking across multiple sets.
        
        Monitors:
        1. Speed degradation - comparing to baseline of initial reps
        2. ROM reduction - tracking changes in range of motion
        3. Form stability - measuring variability in movement patterns
        
        Returns fatigue score and relevant feedback for the user.
        """
        current_time = time.time()
        
        # Get current set number
        current_set = self.sets[exercise_type] + 1 if self.current_set_reps[exercise_type] > 0 else self.sets[exercise_type]
        
        # Check for rest and track it for set detection
        rest_time = current_time - self.last_rep_time[exercise_type]
        
        # If we've done at least one rep and we're not already marked as resting
        if self.last_rep_time[exercise_type] > 0 and not self.is_resting[exercise_type] and rest_time > 5:
            # Start tracking a rest period
            self.is_resting[exercise_type] = True
            self.rest_start_time[exercise_type] = self.last_rep_time[exercise_type]
            
            # Activate rest timer visualization # <-- This toggles the overlay
            self.rest_timer_start[exercise_type] = current_time
            
            # Display rest time to user if it's getting close to a new set
            if rest_time > self.set_pause_threshold * 0.5:  # More than half the threshold
                fatigue_percent = int(self.fatigue_level * 100)
                rest_warning = f"Resting {rest_time:.0f}s - new set in {max(0, self.set_pause_threshold - rest_time):.0f}s. Current fatigue: {fatigue_percent}%"
                return self.fatigue_level, rest_warning
        
        # Check for completion of a set due to extended rest
        if self.is_resting[exercise_type] and rest_time >= self.set_pause_threshold:
            # If we have reps in the current set, complete the set
            if self.current_set_reps[exercise_type] > 0 and len(self.set_history[exercise_type]) == self.sets[exercise_type]:
                # Save the completed set
                self.set_history[exercise_type].append(self.current_set_reps[exercise_type])
                self.set_end_times[exercise_type] = self.last_rep_time[exercise_type]
                
                # Save fatigue metrics for this set before incrementing
                if current_set-1 < len(self.set_fatigue_metrics[exercise_type]):
                    set_metrics = self.set_fatigue_metrics[exercise_type][current_set-1]
                    set_metrics['end_fatigue'] = self.fatigue_level
                    set_metrics['reps'] = self.current_set_reps[exercise_type]
                    
                # Set the transition message and flag
                next_set = self.sets[exercise_type] + 1
                self.show_set_transition[exercise_type] = True
                self.transition_message[exercise_type] = f"Set {next_set} Starting"
                self.transition_start_time[exercise_type] = current_time
                
                # Increment set counter
                self.sets[exercise_type] += 1
                
                # Reset for next set
                self.current_set_reps[exercise_type] = 0
                
                # Reduce fatigue during rest
                if self.fatigue_level > 0:
                    # Exponential decay based on rest time
                    decay_rate = 0.05  # 5% decay per second
                    previous_fatigue = self.fatigue_level
                    self.fatigue_level *= math.exp(-decay_rate * rest_time)
                    self.fatigue_level = max(0.1, self.fatigue_level)  # Keep minimum fatigue at 10%
                    self.prev_fatigue_score = self.fatigue_level
                    
                    # Calculate how much fatigue was reduced
                    fatigue_reduction = previous_fatigue - self.fatigue_level
                    fatigue_percent = int(self.fatigue_level * 100)
                    
                    # Log the set completion due to extended rest
                    print(f"Set {self.sets[exercise_type]} of {exercise_type} completed with {self.set_history[exercise_type][-1]} reps due to {rest_time:.1f}s rest.")
                    print(f"Fatigue reduced by {fatigue_reduction:.2f} ({fatigue_percent}% remaining)")
                    
                    return self.fatigue_level, f"Set {self.sets[exercise_type]} completed! Rest period: {rest_time:.0f}s. Fatigue decreased to {fatigue_percent}%"
        
        # If user has resumed exercise after a rest period, update status
        if self.is_resting[exercise_type] and self.last_rep_time[exercise_type] > self.rest_start_time[exercise_type]:
            # Calculate the rest duration and add it to history
            rest_duration = self.last_rep_time[exercise_type] - self.rest_start_time[exercise_type]
            self.rest_durations[exercise_type].append(rest_duration)
            
            # Reset resting status
            self.is_resting[exercise_type] = False
            
            # If this was a full rest period that didn't get caught by the above logic
            # (can happen if we just missed the threshold), handle it now
            if rest_duration >= self.set_pause_threshold * 0.9 and self.current_set_reps[exercise_type] == 1:
                # Set transition message
                self.show_set_transition[exercise_type] = True
                self.transition_message[exercise_type] = f"Set {current_set} Starting"
                self.transition_start_time[exercise_type] = current_time
                
                print(f"Resumed exercise after {rest_duration:.1f}s rest. Starting set {current_set}.")
        
        # Normal fatigue calculation
        if len(self.rep_timestamps) < min_reps or not self.angle_history[exercise_type]:
            return self.fatigue_level, None

        # Get total reps in this exercise
        total_reps = len(self.angle_history[exercise_type])
        effective_window = min(window_size, total_reps)
        
        # Get timestamps and angles for the latest window
        all_timestamps = [t for t in self.rep_timestamps if t]
        window_timestamps = all_timestamps[-effective_window:]
        window_angles = self.angle_history[exercise_type][-effective_window:]

        # Get or establish baseline speed for this set
        set_key = f"set_{current_set}"
        baseline_reps = min(3, self.current_set_reps[exercise_type])  # Use first 3 reps of set as baseline
        
        # Calculate speed baseline if it doesn't exist for this set
        if set_key not in self.set_baselines[exercise_type]['speed'] and self.current_set_reps[exercise_type] >= baseline_reps:
            # Calculate initial speed baseline from first few reps of the set
            set_start_idx = total_reps - self.current_set_reps[exercise_type]
            baseline_timestamps = all_timestamps[set_start_idx:set_start_idx + baseline_reps]
            if len(baseline_timestamps) > 1:
                baseline_speed = np.mean(np.diff(baseline_timestamps))
                self.set_baselines[exercise_type]['speed'][set_key] = baseline_speed
                print(f"Set {current_set} baseline speed established: {baseline_speed:.2f}s per rep")
                
                # For ROM baseline
                baseline_angles = self.angle_history[exercise_type][set_start_idx:set_start_idx + baseline_reps]
                key_angle = 'right_arm' if exercise_type in ['push_up', 'bicep_curl', 'shoulder_press'] else 'right_leg'
                baseline_angles_values = [a[key_angle] for a in baseline_angles]
                
                # Get appropriate ROM calculation based on exercise type
                if exercise_type in ['push_up', 'squat']:
                    baseline_rom = min(baseline_angles_values)
                else:
                    baseline_rom = max(baseline_angles_values)
                
                self.set_baselines[exercise_type]['rom'][set_key] = baseline_rom
                
                # For variability baseline
                baseline_var = np.std(baseline_angles_values)
                self.set_baselines[exercise_type]['variability'][set_key] = baseline_var
                
                # Initialize metrics tracking for this set
                if len(self.set_fatigue_metrics[exercise_type]) <= current_set-1:
                    self.set_fatigue_metrics[exercise_type].append({
                        'set': current_set,
                        'start_time': self.set_start_times[exercise_type],
                        'baseline_speed': baseline_speed,
                        'baseline_rom': baseline_rom,
                        'baseline_var': baseline_var,
                        'max_speed_change': 0,
                        'max_rom_change': 0,
                        'max_var_change': 0,
                        'start_fatigue': self.fatigue_level,
                        'end_fatigue': 0,
                        'reps': 0
                    })
        
        # Get the actual baseline values for calculations
        baseline_speed = self.set_baselines[exercise_type]['speed'].get(set_key, 0)
        baseline_rom = self.set_baselines[exercise_type]['rom'].get(set_key, 0)
        baseline_var = self.set_baselines[exercise_type]['variability'].get(set_key, 0)
        
        # Calculate current metrics
        rep_times = np.diff(window_timestamps) if len(window_timestamps) > 1 else [0]
        avg_rep_speed = np.mean(rep_times) if len(rep_times) > 0 else 0
        
        # Speed score - how much has speed degraded from baseline
        if baseline_speed > 0:
            speed_change = (avg_rep_speed - baseline_speed) / baseline_speed
            speed_score = min(1.0, max(0.0, speed_change))
            
            # Update max speed change for this set
            if current_set-1 < len(self.set_fatigue_metrics[exercise_type]):
                self.set_fatigue_metrics[exercise_type][current_set-1]['max_speed_change'] = max(
                    self.set_fatigue_metrics[exercise_type][current_set-1]['max_speed_change'],
                    speed_change
                )
        else:
            speed_score = 0.0

        # Variability score - how consistent are the movements
        key_angle = 'right_arm' if exercise_type in ['push_up', 'bicep_curl', 'shoulder_press'] else 'right_leg'
        angles = [rep.get(key_angle, 0) for rep in window_angles]
        variability = np.std(angles) if angles else 0
        
        # Calculate variability change relative to baseline
        if baseline_var > 0:
            var_change = (variability - baseline_var) / baseline_var
            variability_score = min(1.0, max(0.0, var_change))
            
            # Update max variability change for this set
            if current_set-1 < len(self.set_fatigue_metrics[exercise_type]):
                self.set_fatigue_metrics[exercise_type][current_set-1]['max_var_change'] = max(
                    self.set_fatigue_metrics[exercise_type][current_set-1]['max_var_change'],
                    var_change
                )
        else:
            variability_score = min(1.0, variability / 15.0)  # Fallback if baseline not established

        # Calculate ROM changes from ideal or from baseline
        if exercise_type in ['push_up', 'squat']:
            current_rom = min(angles) if angles else 0
            ideal_rom = self.rom_targets[exercise_type]['min']
            
            if baseline_rom > 0:
                rom_change = (baseline_rom - current_rom) / (baseline_rom - 50)
                rom_score = min(1.0, max(0.0, rom_change))
            else:
                rom_score = max(0.0, (ideal_rom - current_rom) / (ideal_rom - 50)) if current_rom < ideal_rom else 0.0
        else:
            current_rom = max(angles) if angles else 0
            ideal_rom = self.rom_targets[exercise_type]['max']
            
            if baseline_rom > 0:
                rom_change = (current_rom - baseline_rom) / (250 - baseline_rom)
                rom_score = min(1.0, max(0.0, rom_change))
            else:
                rom_score = max(0.0, (current_rom - ideal_rom) / (250 - ideal_rom)) if current_rom > ideal_rom else 0.0
        
        # Update max ROM change for this set
        if current_set-1 < len(self.set_fatigue_metrics[exercise_type]):
            rom_change = abs(current_rom - baseline_rom) / baseline_rom if baseline_rom > 0 else 0
            self.set_fatigue_metrics[exercise_type][current_set-1]['max_rom_change'] = max(
                self.set_fatigue_metrics[exercise_type][current_set-1]['max_rom_change'],
                rom_change
            )

        # Special tracking for shoulder press
        if exercise_type == 'shoulder_press':
            self.shoulder_press_metrics['speed_history'].append(avg_rep_speed)
            self.shoulder_press_metrics['rom_history'].append(current_rom)
            self.shoulder_press_metrics['form_history'].append(variability)

        # Calculate combined fatigue score with weighted components
        if exercise_type == 'shoulder_press':
            # Follow the example weighting: speed 50%, ROM 30%, form 20%
            fatigue_score = (0.5 * speed_score + 0.3 * rom_score + 0.2 * variability_score) * weight_factor
        else:
            # Default weighting for other exercises
            fatigue_score = (0.4 * speed_score + 0.3 * rom_score + 0.3 * variability_score) * weight_factor
        
        # Smooth fatigue score with previous value to avoid jumps
        smoothed_fatigue_score = min(1.0, max(0.0, 0.7 * fatigue_score + 0.3 * self.prev_fatigue_score))
        
        # Update fatigue level and trend
        previous_fatigue = self.fatigue_level
        self.fatigue_level = max(self.fatigue_level, smoothed_fatigue_score)
        fatigue_trend = smoothed_fatigue_score - self.prev_fatigue_score
        self.fatigue_trend = 0.7 * fatigue_trend + 0.3 * self.fatigue_trend if self.prev_fatigue_score > 0 else fatigue_trend
        self.prev_fatigue_score = smoothed_fatigue_score
        
        # Find dominant factor in fatigue (what's contributing most)
        fatigue_components = [
            ('speed', speed_score, 'pace is slowing'),
            ('form', variability_score, 'movement is less stable'),
            ('ROM', rom_score, 'range of motion is decreasing')
        ]
        dominant_factor = max(fatigue_components, key=lambda x: x[1])
        
        # Generate appropriate warning message
        warning = None
        fatigue_percent = int(self.fatigue_level * 100)
        
        if self.fatigue_level < 0.2:
            warning = "Feeling fresh—great start!" if self.current_set_reps[exercise_type] <= 3 else None
        elif self.fatigue_level < 0.4:
            if dominant_factor[0] == 'speed':
                warning = f"Pace slowing a bit—stay steady! ({fatigue_percent}% fatigue)"
            elif dominant_factor[0] == 'form':
                warning = f"Slight wobble—keep it smooth! ({fatigue_percent}% fatigue)"
            else:
                warning = f"Range dipping—push a little more! ({fatigue_percent}% fatigue)"
        elif self.fatigue_level < 0.6:
            if dominant_factor[0] == 'speed':
                warning = f"Set {current_set}: Slowing down on {exercise_type}—rest soon? ({fatigue_percent}% fatigue)"
            elif dominant_factor[0] == 'form':
                warning = f"Set {current_set}: Form's getting shaky—focus up! ({fatigue_percent}% fatigue)"
            else:
                warning = f"Set {current_set}: Range dropping—extend fully! ({fatigue_percent}% fatigue)"
        else:
            if dominant_factor[0] == 'speed':
                warning = f"Set {current_set}: High fatigue ({fatigue_percent}%)! Pace way off—rest soon"
            elif dominant_factor[0] == 'form':
                warning = f"Set {current_set}: High fatigue ({fatigue_percent}%)! Form's off—focus on technique"
            else:
                warning = f"Set {current_set}: High fatigue ({fatigue_percent}%)! Range too low—take a break soon"

        # Recovery message if fatigue is trending down
        if self.fatigue_trend < -0.1 and self.fatigue_level < 0.4:
            warning = f"Recovering well on {exercise_type}—nice bounce back! ({fatigue_percent}% fatigue)"

        # Log detailed metrics
        print(f"{exercise_type} Set {current_set} - Speed: {speed_score:.2f}, ROM: {rom_score:.2f}, Form: {variability_score:.2f}")
        print(f"Fatigue: {self.fatigue_level:.2f} ({fatigue_percent}%), Trend: {self.fatigue_trend:+.2f}")
        
        return self.fatigue_level, warning

    def export_fatigue_data(self, filename="fatigue_data.csv"):
        # Enhanced export with set information
        data = {'timestamp': self.rep_timestamps, 'exercise': [], 'angles': [], 'sets': [], 'reps_in_set': []}
        
        # Calculate sets for each timestamp
        rep_set_mapping = {}
        
        for ex in self.set_history.keys():
            set_start_idx = 0
            for set_idx, reps_in_set in enumerate(self.set_history[ex]):
                set_end_idx = set_start_idx + reps_in_set
                for rep_idx in range(set_start_idx, set_end_idx):
                    if rep_idx < len(self.angle_history[ex]):
                        rep_set_mapping[(ex, rep_idx)] = (set_idx + 1, rep_idx - set_start_idx + 1)
                set_start_idx = set_end_idx
        
        # Add current set
        for ex in self.current_set_reps.keys():
            if self.current_set_reps[ex] > 0:
                set_idx = len(self.set_history[ex])
                for rep_idx in range(len(self.angle_history[ex]) - self.current_set_reps[ex], len(self.angle_history[ex])):
                    if rep_idx >= 0:
                        rep_set_mapping[(ex, rep_idx)] = (set_idx + 1, rep_idx - (len(self.angle_history[ex]) - self.current_set_reps[ex]) + 1)
        
        # Build the data
        rep_idx_by_ex = {ex: 0 for ex in self.angle_history.keys()}
        for ex, angles in self.angle_history.items():
            for angle in angles:
                set_info = rep_set_mapping.get((ex, rep_idx_by_ex[ex]), (0, 0))
                data['exercise'].append(ex)
                data['angles'].append(angle)
                data['sets'].append(set_info[0])
                data['reps_in_set'].append(set_info[1])
                rep_idx_by_ex[ex] += 1
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Exercise data exported to {filename}")
        
        # Also export a summary
        summary_data = []
        for ex in self.set_history.keys():
            for set_idx, reps in enumerate(self.set_history[ex]):
                summary_data.append({
                    'exercise': ex,
                    'set': set_idx + 1,
                    'reps': reps,
                    'start_time': self.set_start_times[ex] if set_idx == 0 else 0,  # Only have accurate data for first set
                    'rest_after': self.rest_durations[ex][set_idx] if set_idx < len(self.rest_durations[ex]) else 0
                })
        
        # Add current set if it has reps
        for ex in self.current_set_reps.keys():
            if self.current_set_reps[ex] > 0:
                summary_data.append({
                    'exercise': ex,
                    'set': len(self.set_history[ex]) + 1,
                    'reps': self.current_set_reps[ex],
                    'start_time': self.set_start_times[ex],
                    'rest_after': 0  # Still in progress
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("workout_summary.csv", index=False)
        print(f"Workout summary exported to workout_summary.csv")
        
        return df

    def extract_features(self, landmarks):
        features = []
        if len(landmarks) == len(relevant_landmarks_indices) * 3:
            features.append(calculate_angle(landmarks[0:3], landmarks[6:9], landmarks[12:15]))
            features.append(calculate_angle(landmarks[3:6], landmarks[9:12], landmarks[15:18]))
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))
            features.append(calculate_angle(landmarks[0:3], landmarks[18:21], landmarks[24:27]))
            features.append(calculate_angle(landmarks[3:6], landmarks[21:24], landmarks[27:30]))
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3], landmarks[6:9]))
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6], landmarks[9:12]))
            distances = [
                calculate_distance(landmarks[0:3], landmarks[3:6]),
                calculate_distance(landmarks[18:21], landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30]),
                calculate_distance(landmarks[0:3], landmarks[18:21]),
                calculate_distance(landmarks[3:6], landmarks[21:24]),
                calculate_distance(landmarks[6:9], landmarks[24:27]),
                calculate_distance(landmarks[9:12], landmarks[27:30]),
                calculate_distance(landmarks[12:15], landmarks[0:3]),
                calculate_distance(landmarks[15:18], landmarks[3:6]),
                calculate_distance(landmarks[12:15], landmarks[18:21]),
                calculate_distance(landmarks[15:18], landmarks[21:24])
            ]
            y_distances = [
                calculate_y_distance(landmarks[6:9], landmarks[0:3]),
                calculate_y_distance(landmarks[9:12], landmarks[3:6])
            ]
            normalization_factor = max([d for d in distances if d > 0], default=0.5)
            normalized_distances = [d / normalization_factor if d != -1.0 else d for d in distances]
            normalized_y_distances = [d / normalization_factor if d != -1.0 else d for d in y_distances]
            features.extend(normalized_distances)
            features.extend(normalized_y_distances)
        else:
            features = [-1.0] * 22
        return features

    def preprocess_frame(self, frame, pose):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []
        if results.pose_landmarks:
            for idx in relevant_landmarks_indices:
                landmark = results.pose_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks

    def visualize_angle(self, img, angle, landmark):
        cv2.putText(img, str(int(angle)), tuple(np.multiply(landmark, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def auto_classify_and_count(self):
        # Main layout with proper sections but no tabs
        st.markdown("<h3 style='margin-bottom: 10px; color: #3498DB;'>AI Fitness Trainer</h3>", unsafe_allow_html=True)
        
        # Create a two-column layout with better proportions (video no longer dominates)
        col1, col2 = st.columns([2, 1.2])  # Adjusted ratio to make video smaller and feedback larger
        
        # Header with Stop button in narrow top row
        stop_button = st.button('Stop Exercise', key='stop_button')
        
        # Column 1: Video Feed with enhanced styling
        with col1:
            st.markdown("""
            <div style="
                padding: 8px 0; 
                background: linear-gradient(90deg, #11998e, #38ef7d); 
                border-radius: 8px 8px 0 0;
                text-align: center; 
                font-weight: bold;
                color: white;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0; left: 0; right: 0; bottom: 0;
                    background: linear-gradient(45deg, transparent 49%, rgba(255,255,255,0.1) 50%, transparent 51%);
                    background-size: 20px 20px;
                    z-index: 0;
                "></div>
                <div style="position: relative; z-index: 1;">LIVE CAMERA FEED</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Video frame container with enhanced styling
            st.markdown("""
            <style>
            .stVideo {
                border-radius: 0 0 8px 8px !important;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
                border: 3px solid #2C3E50 !important;
                transition: all 0.3s ease !important;
            }
            .stVideo:hover {
                transform: scale(1.01) !important;
                box-shadow: 0 6px 25px rgba(0,0,0,0.25) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            stframe = st.empty()
            
            # Pro tip with animation instead of plain caption
            st.markdown("""
            <div style="
                margin-top: 10px;
                padding: 10px;
                background: rgba(44, 62, 80, 0.8);
                border-radius: 8px;
                color: white;
                text-align: center;
                font-size: 14px;
                border-left: 4px solid #3498DB;
                animation: slideFadeIn 1s ease;
            ">
                <div style="font-weight: bold; margin-bottom: 5px;">✨ PRO TIP</div>
                Stand approximately 2-3 meters from the camera for best results
            </div>
            
            <style>
            @keyframes slideFadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Column 2: All feedback panels vertically stacked (larger column with better fonts)
        with col2:
            # Form Feedback - increased font sizes and padding
            st.markdown("<h4 style='margin-top: 0px; border-bottom: 2px solid #f0f2f6; padding-bottom: 8px; font-size: 1.2rem;'>Form Feedback</h4>", unsafe_allow_html=True)
            form_feedback_placeholder = st.empty()
            
            # Fatigue Analysis
            st.markdown("<h4 style='border-bottom: 2px solid #f0f2f6; padding-bottom: 8px; margin-top: 18px; font-size: 1.2rem;'>Fatigue Analysis</h4>", unsafe_allow_html=True)
            fatigue_feedback_placeholder = st.empty()
            
            # Exercise Tracking
            st.markdown("<h4 style='border-bottom: 2px solid #f0f2f6; padding-bottom: 8px; margin-top: 18px; font-size: 1.2rem;'>Exercise Tracking</h4>", unsafe_allow_html=True)
            exercise_counter_placeholder = st.empty()

        # Initialize video capture and exercise logic
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error opening webcam.")
            return
        window_size = 30
        landmarks_window = []
        frame_count = 0
        current_prediction = "No prediction yet"
        stages = {'push_up': None, 'squat': None, 'left_bicep_curl': None, 'right_bicep_curl': None, 'shoulder_press': None}
        detector = pm.posture_detector()
        pose = mp.solutions.pose.Pose()
        exercise_name_map = {
            'push_up': 'Push-up',
            'squat': 'Squat',
            'bicep_curl': 'Curl',
            'shoulder_press': 'Press'
        }
        weight_factor = st.sidebar.slider("Weight Intensity (1.0 = Light, 1.5 = Heavy)", 1.0, 1.5, 1.0, 0.1)
        
        # Initialize the current_exercise as None
        current_exercise = None
        exercise_display_name = "Detecting..."
        
        while cap.isOpened():
            if stop_button:
                break
            ret, frame = cap.read()
            if not ret:
                st.error("Error reading frame.")
                break
            current_time = time.time()
            landmarks = self.preprocess_frame(frame, pose)
            if len(landmarks) == len(relevant_landmarks_indices) * 3:
                features = self.extract_features(landmarks)
                if len(features) == 22:
                    landmarks_window.append(features)
            frame_count += 1
            if len(landmarks_window) == window_size:
                landmarks_window_np = np.array(landmarks_window).flatten().reshape(1, -1)
                scaled_landmarks_window = self.scaler.transform(landmarks_window_np)
                scaled_landmarks_window = scaled_landmarks_window.reshape(1, window_size, 22)
                prediction = self.lstm_model.predict(scaled_landmarks_window)
                predicted_class = np.argmax(prediction, axis=1)[0]
                current_prediction = self.exercise_classes[predicted_class]
                landmarks_window = []
                frame_count = 0
            detector.find_person(frame, draw=True)
            landmark_list = detector.find_landmarks(frame, draw=True)
            if len(landmark_list) > 0:
                exercise_type = None
                stage = None
                
                # Improve exercise type detection with confidence threshold
                if current_prediction == 'push-up' or current_prediction == 'push_up':
                    exercise_type = 'push_up'
                    stages['push_up'], self.counters['push_up'] = count_repetition_push_up(
                        detector, frame, landmark_list, stages['push_up'], self.counters['push_up'], self)
                    stage = stages['push_up']
                    current_exercise = 'push_up'
                    exercise_display_name = "PUSH-UP"
                elif current_prediction == 'squat':
                    exercise_type = 'squat'
                    stages['squat'], self.counters['squat'] = count_repetition_squat(
                        detector, frame, landmark_list, stages['squat'], self.counters['squat'], self)
                    stage = stages['squat']
                    current_exercise = 'squat'
                    exercise_display_name = "SQUAT"
                elif current_prediction == 'barbell biceps curl' or current_prediction == 'bicep_curl' or current_prediction == 'bicep curl':
                    exercise_type = 'bicep_curl'
                    
                    # Initialize stages if None
                    if stages['right_bicep_curl'] is None:
                        stages['right_bicep_curl'] = 'none'
                    if stages['left_bicep_curl'] is None:
                        stages['left_bicep_curl'] = 'none'
                    
                    # Add minimal visual guide text as an overlay at bottom of screen
                    h, w, _ = frame.shape
                    guide_text = "BICEP CURL: Extend down, curl up"
                    
                    # Create a semi-transparent background for better readability
                    text_background = np.zeros((40, len(guide_text)*10, 3), dtype=np.uint8)
                    cv2.putText(text_background, guide_text, (10, 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Position at the bottom center of the frame
                    y_pos = h - text_background.shape[0] - 10
                    x_pos = (w - text_background.shape[1]) // 2
                    
                    # Create semi-transparent overlay
                    if y_pos > 0 and x_pos > 0:
                        try:
                            roi = frame[y_pos:y_pos+text_background.shape[0], x_pos:x_pos+text_background.shape[1]]
                            blended = cv2.addWeighted(roi, 0.5, text_background, 0.5, 0)
                            frame[y_pos:y_pos+text_background.shape[0], x_pos:x_pos+text_background.shape[1]] = blended
                        except Exception as e:
                            print(f"Error adding overlay: {e}")
                    
                    # Call rep counter 
                    stages['right_bicep_curl'], stages['left_bicep_curl'], self.counters['bicep_curl'] = count_repetition_bicep_curl(
                        detector, frame, landmark_list, stages['right_bicep_curl'], stages['left_bicep_curl'], 
                        self.counters['bicep_curl'], self)
                    
                    current_exercise = 'bicep_curl'
                    exercise_display_name = "BICEP CURL"
                elif current_prediction == 'shoulder_press' or current_prediction == 'shoulder press':
                    exercise_type = 'shoulder_press'
                    stages['shoulder_press'], self.counters['shoulder_press'] = count_repetition_shoulder_press(
                        detector, frame, landmark_list, stages['shoulder_press'], self.counters['shoulder_press'], self)
                    stage = stages['shoulder_press']
                    current_exercise = 'shoulder_press'
                    exercise_display_name = "SHOULDER PRESS"
                if exercise_type:
                    if current_time - self.last_feedback_time >= self.feedback_interval:
                        new_form_feedback, current_angles, significant_change = generate_feedback(
                            detector, frame, landmark_list, exercise_type, stage, self.feedback_history, self.prev_angles, self
                        )
                        fatigue_score, fatigue_warning = self.detect_fatigue(exercise_type, weight_factor=weight_factor)
                        self.fatigue_level = fatigue_score
                        new_fatigue_feedback = []
                        if fatigue_warning:
                            new_fatigue_feedback.append((fatigue_warning, False))
                            self.fatigue_warnings.append(fatigue_warning)
                        self.form_feedback = new_form_feedback
                        self.prev_form_feedback = new_form_feedback
                        self.last_feedback_time = current_time
                        self.prev_angles = current_angles
                        self.fatigue_feedback = new_fatigue_feedback
                    
                    # Add rest timer overlay if in rest period
                    # if self.show_rest_timer[exercise_type]:
                    #     frame = self.draw_rest_timer_overlay(frame, exercise_type, current_time)
                    
                    # Add set transition overlay if needed
                    if self.show_set_transition[exercise_type]:
                        frame = self.draw_set_transition_overlay(frame, exercise_type, current_time)
                    
                    # Update form feedback display - increased font size
                    with form_feedback_placeholder.container():
                        if self.form_feedback:
                            for i, (msg, is_positive) in enumerate(self.form_feedback[:2]):
                                icon = "✅" if is_positive else "⚠️"
                                message_container = f"""
                                <div style="
                                    padding: 10px; 
                                    border-radius: 5px; 
                                    margin-bottom: 10px;
                                    background-color: {'#E8F5E9' if is_positive else '#FEECEB'};
                                    border-left: 3px solid {'#4CAF50' if is_positive else '#F44336'};
                                    display: flex;
                                    align-items: center;
                                    transition: all 0.3s ease;
                                    font-size: 1rem;
                                ">
                                    <div style="font-size: 18px; margin-right: 10px;">{icon}</div>
                                    <div style="color: #2C3E50; font-weight: 500;">{msg}</div>
                                </div>
                                """
                                st.markdown(message_container, unsafe_allow_html=True)
                        else:
                            st.info("Awaiting form feedback...", icon="⏳")
                    
                    # Update fatigue feedback display with larger fonts
                    with fatigue_feedback_placeholder.container():
                        # Progress bar for fatigue
                        fatigue_percentage = int(self.fatigue_level * 100)
                        fatigue_color = "#4CAF50" if fatigue_percentage < 30 else "#FFB300" if fatigue_percentage < 70 else "#F44336"
                        fatigue_bar = f"""
                        <div style="margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 1rem;">
                                <div style="font-weight: bold; color: #2C3E50;">Fatigue: {fatigue_percentage}%</div>
                                <div style="color: {fatigue_color}; font-weight: bold;">
                                    {"Low" if fatigue_percentage < 30 else "Moderate" if fatigue_percentage < 70 else "High"}
                                </div>
                            </div>
                            <div style="background-color: #ECEFF1; border-radius: 4px; height: 10px; width: 100%;">
                                <div style="background-color: {fatigue_color}; width: {fatigue_percentage}%; 
                                height: 10px; border-radius: 4px; transition: width 0.3s ease;"></div>
                            </div>
                        </div>
                        """
                        st.markdown(fatigue_bar, unsafe_allow_html=True)
                        
                        # Trend indicator - larger font
                        trend_value = self.fatigue_trend
                        trend_icon = "↗️" if trend_value > 0.05 else "➡️" if abs(trend_value) <= 0.05 else "↘️"
                        trend_color = "#F44336" if trend_value > 0.05 else "#4CAF50" if trend_value < -0.05 else "#607D8B"
                        trend_text = f"""
                        <div style="
                            padding: 8px; 
                            border-radius: 4px; 
                            margin-bottom: 10px;
                            background-color: #F5F7FA;
                            display: flex;
                            align-items: center;
                            font-size: 1rem;
                        ">
                            <div style="margin-right: 8px; font-size: 18px;">{trend_icon}</div>
                            <div style="color: {trend_color}; font-weight: bold;">
                                Trend: {trend_value:+.0%}
                            </div>
                        </div>
                        """
                        st.markdown(trend_text, unsafe_allow_html=True)
                        
                        if self.fatigue_feedback:
                            for msg, _ in self.fatigue_feedback:
                                st.warning(msg, icon="⚠️")
                    
                    # Update exercise counter display with improved cards
                    with exercise_counter_placeholder.container():
                        exercise_counts = []
                        for exercise in self.counters.keys():
                            if self.counters[exercise] > 0 or self.sets[exercise] > 0:
                                short_name = exercise_name_map.get(exercise, exercise)
                                
                                # Get set-specific information
                                current_set = self.sets[exercise] + 1 if self.current_set_reps[exercise] > 0 else self.sets[exercise]
                                current_set_reps = self.current_set_reps[exercise]
                                total_reps = self.counters[exercise]
                                
                                # Get rest information
                                is_resting = self.is_resting[exercise]
                                rest_duration = 0
                                if is_resting:
                                    rest_duration = current_time - self.rest_start_time[exercise]
                                    # Calculate time until next set
                                    time_until_new_set = max(0, self.set_pause_threshold - rest_duration)
                                
                                # Add comprehensive tracking information
                                exercise_counts.append((
                                    short_name, 
                                    total_reps, 
                                    current_set,
                                    current_set_reps,
                                    is_resting,
                                    rest_duration,
                                    self.set_pause_threshold if is_resting else 0
                                ))
                        
                        if exercise_counts:
                            # Use simpler Streamlit elements instead of complex HTML
                            for info in exercise_counts:
                                name, total_reps, sets, set_reps, is_resting, rest_duration, pause_threshold = info
                                
                                st.metric(label=f"{name} - Set {sets}", value=f"{set_reps} / {total_reps} Reps")
                                
                                # Display rest information if applicable
                                if is_resting:
                                    rest_progress = min(1.0, rest_duration / pause_threshold)
                                    time_to_new_set = max(0, int(pause_threshold - rest_duration))
                                    
                                    st.info(f"Resting: {int(rest_duration)}s (New set in {time_to_new_set}s)")
                                    st.progress(rest_progress)
                                st.divider() # Add a separator between exercises
                                    
                        else:
                            # More attractive message when no exercises are tracked
                            st.markdown("""
                            <div class="start-exercise-card">
                                <div class="exercise-icon">🏋️</div>
                                <div class="start-message">Start exercising to track reps and sets</div>
                                <div class="instruction">Your progress will appear here</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add custom CSS for better styling
                        st.markdown("""
                        <style>
                        .exercise-tracking-container {
                            display: flex;
                            flex-direction: column;
                            gap: 12px;
                        }
                        .exercise-card {
                            background: linear-gradient(135deg, #2C3E50, #1A2530);
                            border-radius: 8px;
                            padding: 12px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            border-left: 5px solid #3498DB;
                            margin-bottom: 10px;
                            animation: fadeIn 0.5s ease-in-out;
                        }
                        .orange-card { border-left-color: #FF9500; }
                        .green-card { border-left-color: #4CD964; }
                        .blue-card { border-left-color: #007AFF; }
                        .purple-card { border-left-color: #5856D6; }
                        
                        .exercise-header {
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 8px;
                            border-bottom: 1px solid rgba(255,255,255,0.1);
                            padding-bottom: 8px;
                        }
                        .exercise-name {
                            font-weight: bold;
                            font-size: 18px;
                            color: white;
                        }
                        .exercise-set {
                            background-color: rgba(52, 152, 219, 0.2);
                            padding: 3px 8px;
                            border-radius: 4px;
                            font-weight: bold;
                            color: #3498DB;
                        }
                        .exercise-stats {
                            display: flex;
                            justify-content: space-between;
                            margin-bottom: 8px;
                        }
                        .stat-item {
                            display: flex;
                            flex-direction: column;
                        }
                        .stat-label {
                            font-size: 12px;
                            color: #95A5A6;
                        }
                        .stat-value {
                            font-size: 16px;
                            font-weight: bold;
                            color: white;
                        }
                        .rest-timer {
                            background-color: rgba(0,0,0,0.2);
                            padding: 8px;
                            border-radius: 4px;
                            margin-top: 8px;
                        }
                        .rest-header {
                            display: flex;
                            justify-content: space-between;
                            margin-bottom: 5px;
                            font-size: 14px;
                            color: #ECF0F1;
                        }
                        .new-set-time {
                            font-weight: bold;
                        }
                        .red { color: #FF3B30; }
                        .yellow { color: #FFCC00; }
                        .green { color: #4CD964; }
                        
                        .progress-bar-bg {
                            background-color: rgba(255,255,255,0.1);
                            height: 6px;
                            border-radius: 3px;
                            overflow: hidden;
                        }
                        .progress-bar-fill {
                            height: 100%;
                            border-radius: 3px;
                            transition: width 0.3s ease;
                        }
                        .progress-bar-fill.red { background-color: #FF3B30; }
                        .progress-bar-fill.yellow { background-color: #FFCC00; }
                        .progress-bar-fill.green { background-color: #4CD964; }
                        
                        .start-exercise-card {
                            background: linear-gradient(135deg, #2C3E50, #1A2530);
                            border-radius: 8px;
                            padding: 20px;
                            text-align: center;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            animation: pulseAnimation 2s infinite;
                        }
                        .exercise-icon {
                            font-size: 40px;
                            margin-bottom: 10px;
                        }
                        .start-message {
                            font-size: 18px;
                            font-weight: bold;
                            color: white;
                            margin-bottom: 5px;
                        }
                        .instruction {
                            color: #95A5A6;
                            font-size: 14px;
                        }
                        
                        @keyframes fadeIn {
                            from { opacity: 0; transform: translateY(10px); }
                            to { opacity: 1; transform: translateY(0); }
                        }
                        
                        @keyframes pulseAnimation {
                            0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.4); }
                            70% { box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); }
                            100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); }
                        }
                        </style>
                        """, unsafe_allow_html=True)
        
            try:
                height, width, _ = frame.shape
                if height <= 0 or width <= 0:
                    print("Invalid frame dimensions, skipping text rendering.")
                    continue
                
                # Define exercise colors (BGR format) - Brighter, more vibrant colors
                exercise_colors = {
                    'push_up': (0, 165, 255),      # Bright Orange
                    'squat': (0, 255, 127),        # Bright Green
                    'bicep_curl': (255, 0, 127),   # Bright Pink
                    'shoulder_press': (255, 191, 0) # Bright Cyan
                }
                
                # Define gradients for each exercise
                exercise_gradients = {
                    'push_up': [(0, 100, 255), (0, 180, 255)],        # Orange gradient
                    'squat': [(0, 255, 100), (0, 255, 180)],          # Green gradient
                    'bicep_curl': [(255, 0, 100), (255, 100, 200)],   # Pink gradient
                    'shoulder_press': [(255, 150, 0), (255, 200, 50)]  # Cyan gradient
                }
                
                # Add a colorful exercise name banner at the top 
                banner_height = 60
                
                # Create gradient background
                banner = np.zeros((banner_height, width, 3), dtype=np.uint8)
                
                # Get color for current exercise
                base_color = exercise_colors.get(current_exercise, (120, 120, 120))
                
                # Create a gradient banner
                if current_exercise and current_exercise in exercise_gradients:
                    gradient_colors = exercise_gradients[current_exercise]
                    for i in range(width):
                        t = i / width  # Transition parameter [0, 1]
                        r = int((1-t) * gradient_colors[0][0] + t * gradient_colors[1][0])
                        g = int((1-t) * gradient_colors[0][1] + t * gradient_colors[1][1])
                        b = int((1-t) * gradient_colors[0][2] + t * gradient_colors[1][2])
                        banner[:, i] = (b, g, r)
                else:
                    # Default gradient if no exercise selected
                    for i in range(width):
                        t = i / width
                        banner[:, i] = (int(40 + 20*t), int(40 + 20*t), int(40 + 40*t))
                
                # Add a subtle pattern to the banner
                pattern = np.zeros_like(banner)
                for i in range(0, banner_height, 10):
                    cv2.line(pattern, (0, i), (width, i), (255, 255, 255), 1)
                
                # Blend pattern with banner
                banner = cv2.addWeighted(banner, 0.9, pattern, 0.1, 0)
                
                # Add white border at bottom
                cv2.line(banner, (0, banner_height-1), (width, banner_height-1), (255, 255, 255), 2)
                
                # Add detected exercise name with enhanced style
                exercise_text = exercise_display_name if exercise_display_name else "DETECTING..."
                
                # Add glow effect to text
                text_size = cv2.getTextSize(exercise_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
                text_x = (width - text_size[0]) // 2  # Center the text
                
                # Draw text shadow/glow
                for offset in [(2,2), (2,-2), (-2,2), (-2,-2)]:
                    cv2.putText(banner, exercise_text, 
                              (text_x + offset[0], 40 + offset[1]), 
                              cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0, 150), 2, cv2.LINE_AA)
                
                # Draw main text
                cv2.putText(banner, exercise_text, (text_x, 40), 
                          cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Blend the banner with the top of the frame
                frame[:banner_height, :] = banner
                
                # NEW DESIGN: Create a more attractive exercise tracker in top left corner
                box_width = 220  # Smaller width to fit in corner
                box_height = 120  # Initial height - will be adjusted based on content
                margin = 20      # Margin from edges
                
                # Check if we have any exercises to track
                tracked_exercises = []
                for ex in self.counters.keys():
                    if self.counters[ex] > 0 or self.sets[ex] > 0:
                        tracked_exercises.append(ex)
                
                # Adjust height based on number of exercises (min 120px)
                additional_height_per_exercise = 35
                box_height = max(120, 80 + len(tracked_exercises) * additional_height_per_exercise)
                
                # Create modern glass effect background
                tracker_bg = np.zeros((box_height, box_width, 3), dtype=np.uint8)
                
                # Fill with semi-transparent gradient
                for i in range(box_height):
                    t = i / box_height
                    r = int(20 + 10 * (1-t))
                    g = int(25 + 10 * (1-t)) 
                    b = int(30 + 15 * (1-t))
                    tracker_bg[i, :] = (b, g, r)
                
                # Add title with transparent background
                title_height = 30
                title_bg = np.zeros((title_height, box_width, 3), dtype=np.uint8)
                
                # Create gradient title with exercise-specific color if available
                if current_exercise and current_exercise in exercise_gradients:
                    gradient_colors = exercise_gradients[current_exercise]
                    for i in range(box_width):
                        t = i / box_width
                        r = int(t * gradient_colors[0][0] + (1-t) * gradient_colors[1][0])
                        g = int(t * gradient_colors[0][1] + (1-t) * gradient_colors[1][1])
                        b = int(t * gradient_colors[0][2] + (1-t) * gradient_colors[1][2])
                        title_bg[:, i] = (b, g, r)
                else:
                    # Default gradient if no exercise selected
                    for i in range(box_width):
                        t = i / box_width
                        title_bg[:, i] = (int(40 + 40*t), int(40 + 15*t), int(60 + 20*t))
                
                # Add title text with subtle shadow
                title_text = "EXERCISE TRACKER"
                # Shadow effect
                cv2.putText(title_bg, title_text, (11, 21), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0, 150), 1, cv2.LINE_AA)
                # Main text
                cv2.putText(title_bg, title_text, (10, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Add title to tracker background
                tracker_bg[:title_height, :] = title_bg
                
                # Add thin border
                cv2.rectangle(tracker_bg, (0, 0), (box_width-1, box_height-1), (50, 50, 70), 1)
                
                # Add content divider
                cv2.line(tracker_bg, (10, title_height+2), (box_width-10, title_height+2), (255, 255, 255, 80), 1)
                
                # Add exercise tracking content
                y_pos = title_height + 15
                
                if tracked_exercises:
                    for ex in tracked_exercises:
                        # Get data for this exercise
                        display_name = exercise_name_map.get(ex, ex).upper()
                        ex_color = exercise_colors.get(ex, (255, 255, 255))
                        current_set = self.sets[ex] + 1 if self.current_set_reps[ex] > 0 else self.sets[ex]
                        current_set_reps = self.current_set_reps[ex]
                        total_reps = self.counters[ex]
                        
                        # Draw exercise name with color
                        cv2.putText(tracker_bg, display_name, (12, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.55, ex_color, 1, cv2.LINE_AA)
                        
                        # Draw compact info display
                        info_text = f"SET {current_set} • {current_set_reps}/{total_reps} REPS"
                        cv2.putText(tracker_bg, info_text, (12, y_pos + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                        
                        # If this is the current exercise, add highlight
                        if ex == current_exercise:
                            # Draw small indicator
                            cv2.circle(tracker_bg, (5, y_pos-4), 3, ex_color, -1)
                            
                            # If resting, add rest timer on the line below
                            if self.is_resting[ex]:
                                rest_duration = current_time - self.rest_start_time[ex]
                                time_to_new = max(0, self.set_pause_threshold - rest_duration)
                                rest_text = f"REST: {int(rest_duration)}s → NEW: {int(time_to_new)}s"
                                
                                # Draw a small progress bar
                                rest_progress = min(100, int((rest_duration / self.set_pause_threshold) * 100))
                                bar_width = box_width - 24
                                progress_width = int(bar_width * rest_progress / 100)
                                
                                # Progress bar background
                                cv2.rectangle(tracker_bg, (12, y_pos + 25), (12 + bar_width, y_pos + 29), (50, 50, 50), -1)
                                
                                # Progress bar fill - color based on progress
                                if rest_progress < 50:
                                    # Red to yellow
                                    r, g = 255, int(255 * (rest_progress / 50))
                                    bar_color = (0, g, r)
                                else:
                                    # Yellow to green
                                    r, g = int(255 * (1 - (rest_progress - 50) / 50)), 255
                                    bar_color = (0, g, r)
                                
                                # Draw filled progress
                                cv2.rectangle(tracker_bg, (12, y_pos + 25), (12 + progress_width, y_pos + 29), bar_color, -1)
                                
                                # Add rest text above the bar
                                cv2.putText(tracker_bg, rest_text, (12, y_pos + 23), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
                                
                                # Add extra space for rest timer
                                y_pos += 15
                        
                        # Move to next position
                        y_pos += additional_height_per_exercise
                else:
                    # Create a pulsing animation for empty state
                    pulse_value = (np.sin(current_time * 3) + 1) / 2  # Oscillates between 0 and 1
                    text_color = (int(150 + 105 * pulse_value), 
                                  int(150 + 105 * pulse_value), 
                                  int(200 + 55 * pulse_value))
                    
                    cv2.putText(tracker_bg, "No exercises tracked", (20, y_pos + 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
                    cv2.putText(tracker_bg, "Start exercising!", (50, y_pos + 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                
                # Position in top-left corner
                pos_y = banner_height + margin
                pos_x = margin
                
                # Add the tracker to the frame with alpha blending
                # Create a mask for the shape
                mask = np.zeros((box_height, box_width), dtype=np.uint8)
                cv2.rectangle(mask, (0, 0), (box_width, box_height), 255, -1)
                
                # Make sure it fits in the frame
                if pos_y + box_height <= height and pos_x + box_width <= width:
                    roi = frame[pos_y:pos_y+box_height, pos_x:pos_x+box_width]
                    
                    # Blend with transparency
                    alpha = 0.85  # 85% opacity
                    cv2.addWeighted(tracker_bg, alpha, roi, 1-alpha, 0, roi)
                    
                    # Add the ROI back to the frame
                    frame[pos_y:pos_y+box_height, pos_x:pos_x+box_width] = roi
                
                # For bicep curl, show arm positions with animated indicators
                if current_exercise == 'bicep_curl':
                    right_state = stages['right_bicep_curl'].upper() if stages['right_bicep_curl'] else "NONE"
                    left_state = stages['left_bicep_curl'].upper() if stages['left_bicep_curl'] else "NONE"
                    
                    # Create an animated, circular status indicator
                    status_size = 100
                    status_bg = np.zeros((status_size, status_size*2, 3), dtype=np.uint8)
                    
                    # Function to draw animated circle
                    def draw_animated_circle(img, center, state, label, color):
                        # Circle radius based on time for pulsing effect
                        base_radius = 30
                        pulse = np.sin(current_time * 5) * 5 if state == "UP" else 0
                        radius = int(base_radius + pulse)
                        
                        # Draw filled circle with state-based color
                        if state == "DOWN":
                            fill_color = (0, 100, 255)  # Orange
                        elif state == "UP":
                            fill_color = (0, 255, 0)    # Green  
                        else:
                            fill_color = (100, 100, 100)  # Gray
                            
                        cv2.circle(img, center, radius, fill_color, -1)
                        
                        # Draw border
                        cv2.circle(img, center, radius, (255, 255, 255), 2)
                        
                        # Add text
                        cv2.putText(img, state, (center[0]-20, center[1]+5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        
                        # Add label
                        cv2.putText(img, label, (center[0]-12, center[1]+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                    
                    # Draw circles for arm states
                    draw_animated_circle(status_bg, (status_size//2, status_size//2), right_state, "RIGHT", (255, 0, 0))
                    draw_animated_circle(status_bg, (status_size + status_size//2, status_size//2), left_state, "LEFT", (0, 0, 255))
                    
                    # Position at top-right corner
                    x_offset = width - status_bg.shape[1] - 20
                    y_offset = banner_height + 20
                    
                    # Ensure it fits within the frame
                    if (y_offset + status_bg.shape[0] <= height and 
                        x_offset + status_bg.shape[1] <= width):
                        # Create region of interest
                        roi = frame[y_offset:y_offset+status_bg.shape[0], 
                                  x_offset:x_offset+status_bg.shape[1]]
                        # Blend with transparency
                        alpha = 0.85  # 85% opacity for arm state indicators
                        cv2.addWeighted(status_bg, alpha, roi, 1-alpha, 0, roi)
                        frame[y_offset:y_offset+status_bg.shape[0], 
                            x_offset:x_offset+status_bg.shape[1]] = roi
                
            except Exception as e:
                print(f"Error rendering text on frame: {e}")
            
            stframe.image(frame, channels='BGR', use_container_width=True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary section (updated for consistency)
        with st.container():
            st.success("Exercise session completed!", icon="🎉")
            st.markdown("### Exercise Summary", unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown("#### Sets and Repetitions", unsafe_allow_html=True)
                for exercise in self.counters.keys():
                    if self.sets[exercise] > 0 or self.counters[exercise] > 0:
                        st.metric(label=exercise_name_map.get(exercise, exercise),
                                  value=f"{self.sets[exercise]} sets, {self.counters[exercise]} reps")
            with col2:
                st.markdown("#### Latest Form Feedback", unsafe_allow_html=True)
                if self.form_feedback:
                    msg, is_positive = self.form_feedback[0]
                    if is_positive:
                        st.success(msg, icon="✅")
                    else:
                        st.error(msg, icon="⚠️")
                else:
                    st.success("Great form! Keep it up!", icon="👍")
            st.markdown("### Fatigue Analysis", unsafe_allow_html=True)
            st.metric("Peak Fatigue Level", f"{self.fatigue_level:.0%}")
            st.metric("Final Fatigue Trend", f"{self.fatigue_trend:+.0%}")
            if self.fatigue_warnings:
                st.warning("Fatigue Warnings Issued:", icon="⚠️")
                for w in set(self.fatigue_warnings):
                    st.markdown(f"- {w}")
            self.export_fatigue_data()
            if st.button("Start New Session", key="restart_button"):
                st.experimental_rerun()

    def are_hands_joined(self, landmark_list, stop, is_video=False):
        left_wrist = landmark_list[15][1:]
        right_wrist = landmark_list[16][1:]
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        if distance < 30 and not is_video:
            stop = True
        return stop

    def repetitions_counter(self, img, counter):
        # This function is no longer needed for on-video display
        # The counter is now displayed only in the sidebar
        pass
        
    def push_up(self, cap, is_video=False, counter=0, stage=None):
        # If cap is a VideoCapture object, pass it directly; otherwise create a new one
        if isinstance(cap, cv2.VideoCapture):
            self.exercise_method(cap, is_video, count_repetition_push_up, counter=counter, stage=stage)
        else:
            # For backward compatibility - if something other than a VideoCapture was passed
            cap_obj = None  # Pass None so exercise_method will create the camera
            self.exercise_method(cap_obj, is_video, count_repetition_push_up, counter=counter, stage=stage)

    def squat(self, cap, is_video=False, counter=0, stage=None):
        # If cap is a VideoCapture object, pass it directly; otherwise create a new one
        if isinstance(cap, cv2.VideoCapture):
            self.exercise_method(cap, is_video, count_repetition_squat, counter=counter, stage=stage)
        else:
            # For backward compatibility - if something other than a VideoCapture was passed
            cap_obj = None  # Pass None so exercise_method will create the camera
            self.exercise_method(cap_obj, is_video, count_repetition_squat, counter=counter, stage=stage)

    def bicept_curl(self, cap, is_video=False, counter=0, stage_right=None, stage_left=None):
        # If cap is a VideoCapture object, pass it directly; otherwise create a new one
        if isinstance(cap, cv2.VideoCapture):
            self.exercise_method(cap, is_video, count_repetition_bicep_curl, multi_stage=True, counter=counter, stage_right=stage_right, stage_left=stage_left)
        else:
            # For backward compatibility - if something other than a VideoCapture was passed
            cap_obj = None  # Pass None so exercise_method will create the camera
            self.exercise_method(cap_obj, is_video, count_repetition_bicep_curl, multi_stage=True, counter=counter, stage_right=stage_right, stage_left=stage_left)

    def shoulder_press(self, cap, is_video=False, counter=0, stage=None):
        # If cap is a VideoCapture object, pass it directly; otherwise create a new one
        if isinstance(cap, cv2.VideoCapture):
            self.exercise_method(cap, is_video, count_repetition_shoulder_press, counter=counter, stage=stage)
        else:
            # For backward compatibility - if something other than a VideoCapture was passed
            cap_obj = None  # Pass None so exercise_method will create the camera
            self.exercise_method(cap_obj, is_video, count_repetition_shoulder_press, counter=counter, stage=stage)

    def bicep_curl_mode(self):
        """
        A dedicated mode specifically for bicep curl detection and counting.
        This provides enhanced visualization and real-time feedback.
        """
        st.markdown("# Bicep Curl Training Mode")
        st.markdown("Stand facing the camera with your arms visible. Start with arms extended, then curl up and down.")
        
        # Advanced settings in sidebar
        st.sidebar.markdown("### Settings")
        show_debug = st.sidebar.checkbox("Show Debug Information", value=False)
        
        if show_debug:
            st.sidebar.markdown("### Angle Thresholds")
            down_min = st.sidebar.slider("Down Min Angle", 120, 160, 140, 5)
            down_max = st.sidebar.slider("Down Max Angle", 170, 210, 190, 5)
            up_threshold = st.sidebar.slider("Up Angle (< this or > 360-this)", 40, 100, 70, 5)
        
        # Set up camera
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to open webcam. Please check your camera connection.")
            return
            
        detector = pm.posture_detector()
        counter = 0
        stage_right = None 
        stage_left = None
        
        # Instructions and visual guides
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Instructions")
            st.markdown("""
            1. Stand with arms at your sides, palms facing forward
            2. Curl both arms up while keeping elbows close to your body
            3. Lower both arms back to the starting position
            4. Repeat for desired number of repetitions
            5. Join hands in prayer position to exit
            """)
        
        with col2:
            st.markdown("### Current Count")
            count_display = st.markdown(f"## {counter}")
            
            # Add fatigue and set information
            st.markdown("### Stats")
            fatigue_display = st.empty()
            
            # Add rest timer display
            st.markdown("### Rest Timer")
            rest_timer_display = st.empty()
            
            if show_debug:
                st.markdown("### Angle Debug")
                angle_display = st.empty()
        
        # Capture and process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
                
            # Process frame to detect person and landmarks
            img = detector.find_person(frame)
            landmark_list = detector.find_landmarks(img, draw=True)
            
            if len(landmark_list) > 0:
                # Check for stop gesture
                if self.are_hands_joined(landmark_list, stop=False):
                    st.warning("Stop gesture detected. Exiting bicep curl mode.")
                    break
                
                # Get current time
                current_time = time.time()
                
                # Update rest timer display in the right column
                self.display_rest_timer('bicep_curl', rest_timer_display, current_time)
                
                # Add set transition overlay if needed
                if self.show_set_transition['bicep_curl']:
                    img = self.draw_set_transition_overlay(img, 'bicep_curl', current_time)
                
                # Update fatigue display
                fatigue_percent = int(self.fatigue_level * 100)
                current_set = self.sets['bicep_curl'] + 1 if self.current_set_reps['bicep_curl'] > 0 else self.sets['bicep_curl']
                
                # Format the fatigue information
                fatigue_info = f"""
                **Fatigue Level:** {fatigue_percent}%
                
                **Current Set:** {current_set}
                
                **Total Reps:** {self.counters['bicep_curl']}
                """
                fatigue_display.markdown(fatigue_info)
                
                if show_debug:
                    # Override the count_repetition_bicep_curl function with custom thresholds
                    right_arm_angle = detector.find_angle(img, 12, 14, 16)
                    left_arm_angle = detector.find_angle(img, 11, 13, 15)
                    
                    # Draw angle guide
                    h, w, _ = img.shape
                    cv2.putText(img, f"R: {int(right_arm_angle)}°  L: {int(left_arm_angle)}°", 
                              (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Update angle display
                    angle_display.markdown(f"""
                    - Right Arm: **{int(right_arm_angle)}°**
                    - Left Arm: **{int(left_arm_angle)}°**
                    - Right Stage: **{stage_right}**
                    - Left Stage: **{stage_left}**
                    """)
                    
                    # Check for DOWN position with custom thresholds
                    if right_arm_angle > down_min and right_arm_angle < down_max:
                        if stage_right != "down":
                            print(f"Right arm DOWN: {right_arm_angle:.1f}")
                        stage_right = "down"
                    
                    if left_arm_angle > down_min and left_arm_angle < down_max:
                        if stage_left != "down":
                            print(f"Left arm DOWN: {left_arm_angle:.1f}")
                        stage_left = "down"
                    
                    # Check for UP position
                    if stage_right == "down" and stage_left == "down":
                        # Both arms must be in UP position to count a rep
                        up_condition_right = (right_arm_angle < up_threshold) or (right_arm_angle > (360 - up_threshold))
                        up_condition_left = (left_arm_angle < up_threshold) or (left_arm_angle > (360 - up_threshold))
                        
                        if up_condition_right and up_condition_left:
                            stage_right = "up"
                            stage_left = "up"
                            counter += 1
                            
                            # Visual feedback
                            cv2.putText(img, "REP COUNTED!", (int(img.shape[1]/2)-100, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            
                            # Update the counter display
                            count_display.markdown(f"## {counter}")
                            
                            # Update the stats for this exercise
                            if counter != self.counters['bicep_curl']:
                                # We have a new rep, update the counter
                                self.counters['bicep_curl'] = counter
                                # Reset stages so the next rep starts fresh
                                stage_right = None
                                stage_left = None
                    
                    # Display threshold indicators on the screen
                    overlay = img.copy()
                    # Draw DOWN range indicator
                    cv2.rectangle(overlay, (10, 50), (30, 80), (0, 255, 0), -1)
                    cv2.putText(overlay, f"DOWN: {down_min}-{down_max}°", (35, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Draw UP range indicator
                    cv2.rectangle(overlay, (10, 90), (30, 120), (0, 0, 255), -1)
                    cv2.putText(overlay, f"UP: <{up_threshold}° or >{360-up_threshold}°", (35, 110), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Add the overlay with transparency
                    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                else:
                    # Call the standard repetition counter function
                    stage_right, stage_left, counter = count_repetition_bicep_curl(
                        detector, img, landmark_list, stage_right, stage_left, counter, self)
                    
                    # Update counter display
                    count_display.markdown(f"## {counter}")
                    
                    # Update the stats for this exercise
                    if counter != self.counters['bicep_curl']:
                        # We have a new rep, update the counter
                        self.counters['bicep_curl'] = counter
                        # Reset stages so the next rep starts fresh
                        stage_right = None
                        stage_left = None
                
                # Process form feedback (if sufficient time has passed)
                if current_time - self.last_feedback_time >= self.feedback_interval:
                    form_feedback, self.prev_angles, significant_change = generate_feedback(
                        detector, img, landmark_list, 'bicep_curl', stage_right, 
                        self.feedback_history, self.prev_angles, self
                    )
                    
                    # Only update feedback if there's significant change or it's been a while
                    if significant_change or current_time - self.last_feedback_time > 10:
                        self.form_feedback = form_feedback
                        self.last_feedback_time = current_time
                
                # Add overlay with instructions at the bottom of the frame
                h, w, _ = img.shape
                instruction_overlay = np.zeros((60, w, 3), dtype=np.uint8)
                cv2.putText(instruction_overlay, "DOWN: Extend arms | UP: Curl arms | JOIN HANDS: Exit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add instruction overlay to bottom of frame
                img_with_overlay = np.vstack([img, instruction_overlay])
                
                # Display the frame
                stframe.image(img_with_overlay, channels='BGR', use_column_width=True)
                
                # Update form feedback in a more appealing way
                if len(self.form_feedback) > 0:
                    st.markdown("### Form Feedback")
                    for i, feedback in enumerate(self.form_feedback[:3]):  # Show top 3 feedback items
                        st.markdown(f"- {feedback}")
            else:
                # If no person detected
                cv2.putText(img, "No person detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2, cv2.LINE_AA)
                stframe.image(img, channels='BGR', use_column_width=True)
        
        cap.release()
        
        # Summary
        st.success(f"Bicep Curl session completed! Total repetitions: {counter}")
        if counter > 0:
            st.balloons()
        
        # Option to restart
        if st.button("Start New Session"):
            st.experimental_rerun()

    def exercise_method(self, cap, is_video, count_repetition_function, multi_stage=False, counter=0, stage=None, stage_right=None, stage_left=None):
        # Create a placeholder for rest timer display
        rest_timer_display = st.empty()
        
        if is_video:
            stframe = st.empty()
            detector = pm.posture_detector()
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_time = 1 / original_fps
            frame_count = 0
            start_time = time.time()
            last_update_time = start_time
            update_interval = 0.1
            while cap.isOpened():
                current_time = time.time()
                elapsed_time = current_time - start_time
                target_frame = int(elapsed_time * original_fps)
                while frame_count < target_frame:
                    ret, frame = cap.read()
                    if not ret:
                        return
                    frame_count += 1
                    if frame_count == target_frame:
                        img = detector.find_person(frame)
                        landmark_list = detector.find_landmarks(img, draw=False)
                        if len(landmark_list) != 0:
                            exercise_type = None
                            if count_repetition_function.__name__ == 'count_repetition_push_up':
                                exercise_type = 'push_up'
                            elif count_repetition_function.__name__ == 'count_repetition_squat':
                                exercise_type = 'squat'
                            elif count_repetition_function.__name__ == 'count_repetition_bicep_curl':
                                exercise_type = 'bicep_curl'
                            elif count_repetition_function.__name__ == 'count_repetition_shoulder_press':
                                exercise_type = 'shoulder_press'
                            if exercise_type and current_time - self.last_feedback_time >= self.feedback_interval:
                                stage_to_pass = stage if exercise_type in ['push_up', 'squat', 'shoulder_press'] else None
                                new_feedback, current_angles, significant_change = generate_feedback(
                                    detector, img, landmark_list, exercise_type, stage_to_pass, self.feedback_history, self.prev_angles, self
                                )
                                if significant_change or new_feedback != self.prev_form_feedback:
                                    self.form_feedback = new_feedback
                                    self.prev_form_feedback = new_feedback
                                    self.last_feedback_time = current_time
                                    self.prev_angles = current_angles
                                    
                                # Also run fatigue detection
                                fatigue_score, fatigue_warning = self.detect_fatigue(exercise_type, weight_factor=1.0)
                                self.fatigue_level = fatigue_score
                            
                            # Update rest timer display in the sidebar instead of overlay
                            if exercise_type:
                                self.display_rest_timer(exercise_type, rest_timer_display, current_time)
                                
                            # Add set transition overlay if needed
                            if exercise_type and self.show_set_transition[exercise_type]:
                                img = self.draw_set_transition_overlay(img, exercise_type, current_time)
                                
                            if multi_stage:
                                stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                            else:
                                stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)
                        self.repetitions_counter(img, counter)
                if current_time - last_update_time >= update_interval:
                    stframe.image(img, channels='BGR', use_container_width=True)
                    last_update_time = current_time
                time.sleep(0.001)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            stframe = st.empty()
            # Check if cap is already initialized; if not, create a new VideoCapture
            cap_owned_locally = False
            if cap is None:
                cap = cv2.VideoCapture(0)
                cap_owned_locally = True
                
            detector = pm.posture_detector()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = time.time()
                img = detector.find_person(frame)
                landmark_list = detector.find_landmarks(img, draw=False)
                if len(landmark_list) != 0:
                    exercise_type = None
                    if count_repetition_function.__name__ == 'count_repetition_push_up':
                        exercise_type = 'push_up'
                    elif count_repetition_function.__name__ == 'count_repetition_squat':
                        exercise_type = 'squat'
                    elif count_repetition_function.__name__ == 'count_repetition_bicep_curl':
                        exercise_type = 'bicep_curl'
                    elif count_repetition_function.__name__ == 'count_repetition_shoulder_press':
                        exercise_type = 'shoulder_press'
                    
                    if exercise_type and current_time - self.last_feedback_time >= self.feedback_interval:
                        stage_to_pass = stage if exercise_type in ['push_up', 'squat', 'shoulder_press'] else None
                        new_feedback, current_angles, significant_change = generate_feedback(
                            detector, img, landmark_list, exercise_type, stage_to_pass, self.feedback_history, self.prev_angles, self
                        )
                        
                        if significant_change or new_feedback != self.prev_form_feedback:
                            self.form_feedback = new_feedback
                            self.prev_form_feedback = new_feedback
                            self.last_feedback_time = current_time
                            self.prev_angles = current_angles
                            
                        # Run fatigue detection
                        fatigue_score, fatigue_warning = self.detect_fatigue(exercise_type, weight_factor=1.0)
                        self.fatigue_level = fatigue_score
                        if fatigue_warning:
                            print(f"Fatigue warning: {fatigue_warning}")
                    
                    # Update rest timer display in the sidebar instead of overlay
                    if exercise_type:
                        self.display_rest_timer(exercise_type, rest_timer_display, current_time)
                        
                    # Add set transition overlay if needed
                    if exercise_type and self.show_set_transition[exercise_type]:
                        img = self.draw_set_transition_overlay(img, exercise_type, current_time)
                        
                    if multi_stage:
                        stage_right, stage_left, counter = count_repetition_function(detector, img, landmark_list, stage_right, stage_left, counter, self)
                    else:
                        stage, counter = count_repetition_function(detector, img, landmark_list, stage, counter, self)
                self.repetitions_counter(img, counter)
                stframe.image(img, channels='BGR', use_container_width=True)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Only release the VideoCapture if we created it in this method
            if cap_owned_locally:
                cap.release()
        return stage, counter

    def push_up_mode(self):
        """
        A dedicated mode specifically for push-up detection and counting.
        This provides enhanced visualization and real-time feedback.
        """
        st.markdown("# Push-up Training Mode")
        st.markdown("Position yourself in a plank position with your body visible to the camera. Lower your body and push back up.")
        
        # Advanced settings in sidebar
        st.sidebar.markdown("### Settings")
        show_debug = st.sidebar.checkbox("Show Debug Information", value=False)
        
        if show_debug:
            st.sidebar.markdown("### Angle Thresholds")
            up_threshold = st.sidebar.slider("Up Threshold", 230, 270, 250, 5)
            down_threshold = st.sidebar.slider("Down Threshold", 210, 250, 240, 5)
        
        # Set up camera
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to open webcam. Please check your camera connection.")
            return
            
        detector = pm.posture_detector()
        counter = 0
        stage = None
        
        # Instructions and visual guides
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Instructions")
            st.markdown("""
            1. Start in a high plank position with arms straight
            2. Lower your body until your elbows are at approximately 90 degrees
            3. Push back up to the starting position
            4. Keep your core tight and back straight
            5. Join hands in prayer position to exit
            """)
        
        with col2:
            st.markdown("### Current Count")
            count_display = st.markdown(f"## {counter}")
            
            # Add fatigue and set information
            st.markdown("### Stats")
            fatigue_display = st.empty()
            
            # Add rest timer display
            st.markdown("### Rest Timer")
            rest_timer_display = st.empty()
            
            if show_debug:
                st.markdown("### Angle Debug")
                angle_display = st.empty()
        
        # Capture and process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
                
            # Process frame to detect person and landmarks
            img = detector.find_person(frame)
            landmark_list = detector.find_landmarks(img, draw=True)
            
            if len(landmark_list) > 0:
                # Check for stop gesture
                if self.are_hands_joined(landmark_list, stop=False):
                    st.warning("Stop gesture detected. Exiting push-up mode.")
                    break
                
                # Calculate angles for arms
                right_arm_angle = detector.find_angle(img, 12, 14, 16)
                left_arm_angle = detector.find_angle(img, 11, 13, 15)
                avg_arm_angle = (right_arm_angle + left_arm_angle) / 2
                
                # Get visual landmarks
                right_shoulder = landmark_list[12][1:]
                left_shoulder = landmark_list[11][1:]
                
                # Visualize angles
                self.visualize_angle(img, right_arm_angle, right_shoulder)
                self.visualize_angle(img, left_arm_angle, left_shoulder)
                
                # Get current time
                current_time = time.time()
                
                # Update rest timer display in the right column instead of overlay
                self.display_rest_timer('push_up', rest_timer_display, current_time)
                
                # Add set transition overlay if needed
                if self.show_set_transition['push_up']:
                    img = self.draw_set_transition_overlay(img, 'push_up', current_time)
                
                if show_debug:
                    # Display angle debug info
                    angle_info = f"""
                    Right Arm: {right_arm_angle:.1f}°
                    Left Arm: {left_arm_angle:.1f}°
                    Average: {avg_arm_angle:.1f}°
                    """
                    angle_display.markdown(angle_info)
                    
                # Update fatigue display
                fatigue_percent = int(self.fatigue_level * 100)
                current_set = self.sets['push_up'] + 1 if self.current_set_reps['push_up'] > 0 else self.sets['push_up']
                
                # Format the fatigue information
                fatigue_info = f"""
                **Fatigue Level:** {fatigue_percent}%
                
                **Current Set:** {current_set}
                
                **Total Reps:** {self.counters['push_up']}
                """
                fatigue_display.markdown(fatigue_info)
                
                # Process form feedback (if sufficient time has passed)
                if current_time - self.last_feedback_time >= self.feedback_interval:
                    form_feedback, self.prev_angles, significant_change = generate_feedback(
                        detector, img, landmark_list, 'push_up', stage, 
                        self.feedback_history, self.prev_angles, self
                    )
                    
                    # Only update feedback if there's significant change or it's been a while
                    if significant_change or current_time - self.last_feedback_time > 10:
                        self.form_feedback = form_feedback
                        self.last_feedback_time = current_time
                
                # Process push-up rep counting
                stage, counter = count_repetition_push_up(detector, img, landmark_list, stage, counter, self)
                count_display.markdown(f"## {counter}")
                
                # Update the stats for this exercise
                if counter != self.counters['push_up']:
                    # We have a new rep, update the counter
                    self.counters['push_up'] = counter
                    # Reset the stage so the next rep starts fresh
                    stage = None
            
            # Display the frame with overlay
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img, channels="RGB", use_column_width=True)
            
            # Update form feedback in a more appealing way
            if len(self.form_feedback) > 0:
                st.markdown("### Form Feedback")
                for i, feedback in enumerate(self.form_feedback[:3]):  # Show top 3 feedback items
                    st.markdown(f"- {feedback}")
        
        # Close the camera
        cap.release()

    def squat_mode(self):
        """
        A dedicated mode specifically for squat detection and counting.
        This provides enhanced visualization and real-time feedback.
        """
        st.markdown("# Squat Training Mode")
        st.markdown("Stand sideways to the camera with your full body visible. Bend your knees and lower your body, then return to standing.")
        
        # Advanced settings in sidebar
        st.sidebar.markdown("### Settings")
        show_debug = st.sidebar.checkbox("Show Debug Information", value=False)
        
        if show_debug:
            st.sidebar.markdown("### Angle Thresholds")
            up_threshold = st.sidebar.slider("Up Threshold", 140, 160, 150, 5)
            down_min = st.sidebar.slider("Down Min Angle", 180, 200, 190, 5)
        
        # Set up camera
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to open webcam. Please check your camera connection.")
            return
            
        detector = pm.posture_detector()
        counter = 0
        stage = None
        
        # Instructions and visual guides
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Instructions")
            st.markdown("""
            1. Stand with feet shoulder-width apart, sideways to the camera
            2. Lower your body by bending your knees and pushing hips back
            3. Go down until thighs are parallel to the floor (or as low as comfortable)
            4. Push through heels to return to starting position
            5. Join hands in prayer position to exit
            """)
        
        with col2:
            st.markdown("### Current Count")
            count_display = st.markdown(f"## {counter}")
            
            # Add fatigue and set information
            st.markdown("### Stats")
            fatigue_display = st.empty()
            
            # Add rest timer display
            st.markdown("### Rest Timer")
            rest_timer_display = st.empty()
            
            if show_debug:
                st.markdown("### Angle Debug")
                angle_display = st.empty()
        
        # Capture and process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
                
            # Process frame to detect person and landmarks
            img = detector.find_person(frame)
            landmark_list = detector.find_landmarks(img, draw=True)
            
            if len(landmark_list) > 0:
                # Check for stop gesture
                if self.are_hands_joined(landmark_list, stop=False):
                    st.warning("Stop gesture detected. Exiting squat mode.")
                    break
                
                # Calculate leg angles
                right_leg_angle = detector.find_angle(img, 24, 26, 28)
                left_leg_angle = detector.find_angle(img, 23, 25, 27)
                
                # Visualize the right leg angle
                right_knee = landmark_list[26][1:]
                self.visualize_angle(img, right_leg_angle, right_knee)
                
                # Get current time
                current_time = time.time()
                
                # Update rest timer display in the right column instead of overlay
                self.display_rest_timer('squat', rest_timer_display, current_time)
                
                # Add set transition overlay if needed
                if self.show_set_transition['squat']:
                    img = self.draw_set_transition_overlay(img, 'squat', current_time)
                
                if show_debug:
                    # Display angle debug info
                    angle_info = f"""
                    Right Leg: {right_leg_angle:.1f}°
                    Left Leg: {left_leg_angle:.1f}°
                    """
                    angle_display.markdown(angle_info)
                
                # Update fatigue display
                fatigue_percent = int(self.fatigue_level * 100)
                current_set = self.sets['squat'] + 1 if self.current_set_reps['squat'] > 0 else self.sets['squat']
                
                # Format the fatigue information
                fatigue_info = f"""
                **Fatigue Level:** {fatigue_percent}%
                
                **Current Set:** {current_set}
                
                **Total Reps:** {self.counters['squat']}
                """
                fatigue_display.markdown(fatigue_info)
                
                # Process form feedback (if sufficient time has passed)
                if current_time - self.last_feedback_time >= self.feedback_interval:
                    form_feedback, self.prev_angles, significant_change = generate_feedback(
                        detector, img, landmark_list, 'squat', stage, 
                        self.feedback_history, self.prev_angles, self
                    )
                    
                    # Only update feedback if there's significant change or it's been a while
                    if significant_change or current_time - self.last_feedback_time > 10:
                        self.form_feedback = form_feedback
                        self.last_feedback_time = current_time
                
                # Process squat rep counting
                stage, counter = count_repetition_squat(detector, img, landmark_list, stage, counter, self)
                count_display.markdown(f"## {counter}")
                
                # Update the stats for this exercise
                if counter != self.counters['squat']:
                    # We have a new rep, update the counter
                    self.counters['squat'] = counter
                    # Reset the stage so the next rep starts fresh
                    stage = None
            
            # Display the frame with overlay
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img, channels="RGB", use_column_width=True)
            
            # Update form feedback in a more appealing way
            if len(self.form_feedback) > 0:
                st.markdown("### Form Feedback")
                for i, feedback in enumerate(self.form_feedback[:3]):  # Show top 3 feedback items
                    st.markdown(f"- {feedback}")
        
        # Close the camera
        cap.release()

    def shoulder_press_mode(self):
        """
        A dedicated mode specifically for shoulder press detection and counting.
        This provides enhanced visualization and real-time feedback.
        """
        st.markdown("# Shoulder Press Training Mode")
        st.markdown("Stand facing the camera with your arms visible. Start with arms at shoulder level, then press upward.")
        
        # Advanced settings in sidebar
        st.sidebar.markdown("### Settings")
        show_debug = st.sidebar.checkbox("Show Debug Information", value=False)
        
        if show_debug:
            st.sidebar.markdown("### Angle Thresholds")
            down_min = st.sidebar.slider("Down Min Angle", 230, 270, 250, 5)
            down_max = st.sidebar.slider("Down Max Angle", 270, 290, 280, 5)
            up_threshold = st.sidebar.slider("Up Max Angle", 170, 220, 210, 5)
        
        # Set up camera
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to open webcam. Please check your camera connection.")
            return
            
        detector = pm.posture_detector()
        counter = 0
        stage = None
        
        # Instructions and visual guides
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Instructions")
            st.markdown("""
            1. Stand with weights at shoulder level, elbows bent
            2. Press weights directly overhead until arms are fully extended
            3. Lower weights back to shoulder level
            4. Keep core engaged and avoid arching your back
            5. Join hands in prayer position to exit
            """)
        
        with col2:
            st.markdown("### Current Count")
            count_display = st.markdown(f"## {counter}")
            
            st.markdown("### Stats")
            fatigue_display = st.empty()
            
            # Add rest timer display
            st.markdown("### Rest Timer")
            rest_timer_display = st.empty()
            
            if show_debug:
                st.markdown("### Angle Debug")
                angle_display = st.empty()
        
        # Capture and process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
                
            # Process frame to detect person and landmarks
            img = detector.find_person(frame)
            landmark_list = detector.find_landmarks(img, draw=True)
            
            if len(landmark_list) > 0:
                # Check for stop gesture
                if self.are_hands_joined(landmark_list, stop=False):
                    st.warning("Stop gesture detected. Exiting shoulder press mode.")
                    break
                
                # Calculate arm angles
                right_arm_angle = detector.find_angle(img, 12, 14, 16)
                left_arm_angle = detector.find_angle(img, 11, 13, 15)
                
                # Visualize the angles
                right_elbow = landmark_list[14][1:]
                left_elbow = landmark_list[13][1:]
                self.visualize_angle(img, right_arm_angle, right_elbow)
                self.visualize_angle(img, left_arm_angle, left_elbow)
                
                # Get current time
                current_time = time.time()
                
                # Update rest timer display in the right column instead of overlay
                self.display_rest_timer('shoulder_press', rest_timer_display, current_time)
                
                # Add set transition overlay if needed
                if self.show_set_transition['shoulder_press']:
                    img = self.draw_set_transition_overlay(img, 'shoulder_press', current_time)
                
                if show_debug:
                    # Display angle debug info
                    angle_info = f"""
                    Right Arm: {right_arm_angle:.1f}°
                    Left Arm: {left_arm_angle:.1f}°
                    """
                    angle_display.markdown(angle_info)
                
                # Update fatigue display
                fatigue_percent = int(self.fatigue_level * 100)
                current_set = self.sets['shoulder_press'] + 1 if self.current_set_reps['shoulder_press'] > 0 else self.sets['shoulder_press']
                
                # Format the fatigue information
                fatigue_info = f"""
                **Fatigue Level:** {fatigue_percent}%
                
                **Current Set:** {current_set}
                
                **Total Reps:** {self.counters['shoulder_press']}
                """
                fatigue_display.markdown(fatigue_info)
                
                # Process form feedback (if sufficient time has passed)
                if current_time - self.last_feedback_time >= self.feedback_interval:
                    form_feedback, self.prev_angles, significant_change = generate_feedback(
                        detector, img, landmark_list, 'shoulder_press', stage, 
                        self.feedback_history, self.prev_angles, self
                    )
                    
                    # Only update feedback if there's significant change or it's been a while
                    if significant_change or current_time - self.last_feedback_time > 10:
                        self.form_feedback = form_feedback
                        self.last_feedback_time = current_time
                
                # Process shoulder press rep counting
                stage, counter = count_repetition_shoulder_press(detector, img, landmark_list, stage, counter, self)
                count_display.markdown(f"## {counter}")
                
                # Update the stats for this exercise
                if counter != self.counters['shoulder_press']:
                    # We have a new rep, update the counter
                    self.counters['shoulder_press'] = counter
                    # Reset the stage so the next rep starts fresh
                    stage = None
            
            # Display the frame with overlay
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img, channels="RGB", use_column_width=True)
            
            # Update form feedback in a more appealing way
            if len(self.form_feedback) > 0:
                st.markdown("### Form Feedback")
                for i, feedback in enumerate(self.form_feedback[:3]):  # Show top 3 feedback items
                    st.markdown(f"- {feedback}")
        
        # Close the camera
        cap.release()

    def draw_set_transition_overlay(self, img, exercise_type, current_time):
        """
        Draws a set transition message when moving from one set to another.
        """
        if not self.show_set_transition[exercise_type]:
            return img
        
        # Check if we should still show the transition
        time_elapsed = current_time - self.transition_start_time[exercise_type]
        if time_elapsed > self.transition_duration:
            self.show_set_transition[exercise_type] = False
            return img
        
        # Create overlay with transition message
        h, w, _ = img.shape
        
        # Determine opacity based on time (fade in/out effect)
        if time_elapsed < 0.5:
            # Fade in
            alpha = time_elapsed / 0.5
        elif time_elapsed > (self.transition_duration - 0.5):
            # Fade out
            alpha = (self.transition_duration - time_elapsed) / 0.5
        else:
            # Full opacity
            alpha = 1.0
            
        # Create a dark semi-transparent overlay for the entire image
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5 * alpha, img, 1, 0, img)
        
        # Get current set
        current_set = self.sets[exercise_type]
        
        # Add animated text in the center
        # Make text size pulse for attention
        pulse_factor = 1.0 + 0.2 * math.sin(time_elapsed * 4)
        font_size = 1.5 * pulse_factor
        
        # Draw main message (e.g., "Set 2 Starting")
        text = self.transition_message[exercise_type]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_size, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h - text_size[1]) // 2
        
        # Draw shadow/outline for better visibility
        outline_color = (0, 0, 0)
        for offset_x, offset_y in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            cv2.putText(img, text, (text_x + offset_x, text_y + offset_y), 
                      cv2.FONT_HERSHEY_DUPLEX, font_size, outline_color, 2, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(img, text, (text_x, text_y), 
                  cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add fatigue information
        fatigue_percent = int(self.fatigue_level * 100)
        fatigue_text = f"Fatigue Level: {fatigue_percent}%"
        fatigue_size = cv2.getTextSize(fatigue_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        fatigue_x = (w - fatigue_size[0]) // 2
        fatigue_y = text_y + text_size[1] + 40
        
        cv2.putText(img, fatigue_text, (fatigue_x, fatigue_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show previous set info if available
        if current_set > 0 and len(self.set_history[exercise_type]) > 0:
            prev_reps = self.set_history[exercise_type][-1]
            prev_set_text = f"Previous set: {prev_reps} reps"
            prev_set_size = cv2.getTextSize(prev_set_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            prev_set_x = (w - prev_set_size[0]) // 2
            prev_set_y = fatigue_y + 40
            
            cv2.putText(img, prev_set_text, (prev_set_x, prev_set_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return img

    # Define a helper method for displaying rest timer at the top of the Exercise class
    def display_rest_timer(self, exercise_type, rest_timer_display, current_time):
        """
        Display a properly formatted rest timer in a Streamlit container
        """
        if self.show_rest_timer[exercise_type]:
            rest_elapsed = current_time - self.rest_timer_start[exercise_type]
            time_until_new_set = max(0, self.set_pause_threshold - rest_elapsed)
            progress_percent = min(100, int((rest_elapsed / self.set_pause_threshold) * 100))
            
            # Create a styled rest timer display
            rest_timer_display.markdown(f"**REST PERIOD** - {int(rest_elapsed)}s elapsed")
            rest_timer_display.markdown(f"⏱️ **New set in: {int(time_until_new_set)}s**")
            
            # Create a progress bar
            progress_bar_html = f"""
            <div style="width:100%; background-color:#e0e0e0; height:10px; border-radius:5px; margin:10px 0;">
                <div style="width:{progress_percent}%; background-color:{'#ff9d00' if progress_percent > 80 else '#2196F3'}; height:10px; border-radius:5px;"></div>
            </div>
            <p style="text-align:center; font-size:0.8em;">{progress_percent}% complete</p>
            """
            rest_timer_display.markdown(progress_bar_html, unsafe_allow_html=True)
        else:
            rest_timer_display.markdown("*No rest period active*")

if __name__ == "__main__":
    exercise = Exercise()
    
    # Add mode selection
    st.sidebar.markdown("# Exercise AI Trainer")
    app_mode = st.sidebar.selectbox(
        "Choose Mode", 
        ["Auto Detection", "Bicep Curl Mode", "Push-up", "Squat", "Shoulder Press"]
    )
    
    if app_mode == "Auto Detection":
        exercise.auto_classify_and_count()
    elif app_mode == "Bicep Curl Mode":
        exercise.bicep_curl_mode()
    elif app_mode == "Push-up":
        exercise.push_up(None)
    elif app_mode == "Squat":
        exercise.squat(None)
    elif app_mode == "Shoulder Press":
        exercise.shoulder_press(None)
