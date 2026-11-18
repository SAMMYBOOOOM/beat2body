"""The shared skeleton (COCO-17) and the joint mappings of the three pose tools."""

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO17_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
# joint order after a horizontal flip (left <-> right)
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# OpenPose BODY-18 order (DWPose body output and the OpenPose PyTorch port)
OPENPOSE18_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
OPENPOSE18_TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

# official OpenPose BODY_25
BODY25_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
    "left_wrist", "mid_hip", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
    "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe", "left_small_toe",
    "left_heel", "right_big_toe", "right_small_toe", "right_heel",
]
BODY25_TO_COCO17 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]

# MediaPipe Pose 33 landmarks
MEDIAPIPE33_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
MEDIAPIPE33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Default "joint is reliable" confidence threshold per tool; the scores are on different scales.
CONF_THRESHOLD = {"dwpose": 0.3, "mediapipe": 0.5, "openpose": 0.1}
