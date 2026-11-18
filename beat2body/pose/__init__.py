"""4.3 Pose estimation: one module per tool, all writing the same raw format.

    python -m beat2body pose dwpose    --categories la_hiphop house --preview 2
    python -m beat2body pose mediapipe --device cpu
    python -m beat2body pose openpose  --backend official

Output per video: <data>/keypoints/<category>/<video_id>_<tool>.npz with
    keypoints (T, 17, 3)  COCO-17 x, y (pixels), confidence; NaN where no person was found
    native    (T, J, 3)   the tool's own joint layout
Next step (4.4): python -m beat2body clean --tool <tool>
"""
from beat2body.pose import dwpose, mediapipe, openpose

TOOLS = {"dwpose": dwpose, "mediapipe": mediapipe, "openpose": openpose}
