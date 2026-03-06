"""
MediaPipe Pose -> MotionScript joint converter.
Uses the new MediaPipe Tasks API (matches realtime_test.ipynb).

Run:
    python mediapipe_to_motionscript.py --source 0 --output my_motion.pt
    python mediapipe_to_motionscript.py --source myvideo.mp4 --output my_motion.pt

Press 'q' to stop. Saves (N_frames, 26, 3) tensor.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python import vision
import numpy as np
import torch
import argparse

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = np.copy(rgb_image)
    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
    return annotated


def landmarks_to_motionscript_joints(pose_landmarks):
    """
    Convert one frame of MediaPipe landmarks to MotionScript joint coords.
    pose_landmarks: list of NormalizedLandmark (33 landmarks)
    Returns np.ndarray of shape (26, 3)
    """
    lm = np.array([[l.x, l.y, l.z] for l in pose_landmarks])

    # Flip y so that up is positive (MediaPipe y increases downward)
    lm[:, 1] = -lm[:, 1]

    left_hip     = lm[23]
    right_hip    = lm[24]
    left_sh      = lm[11]
    right_sh     = lm[12]

    pelvis       = (left_hip + right_hip) / 2
    neck         = (left_sh + right_sh) / 2
    spine1       = pelvis + (neck - pelvis) * 0.33
    spine2       = pelvis + (neck - pelvis) * 0.66
    spine3       = neck - np.array([0, 0.05, 0])
    left_collar  = (neck + left_sh) / 2
    right_collar = (neck + right_sh) / 2
    head         = (lm[7] + lm[8]) / 2  # ears

    joints = np.array([
        pelvis,        # 0  pelvis
        left_hip,      # 1  left_hip
        right_hip,     # 2  right_hip
        spine1,        # 3  spine1
        lm[25],        # 4  left_knee
        lm[26],        # 5  right_knee
        spine2,        # 6  spine2
        lm[27],        # 7  left_ankle
        lm[28],        # 8  right_ankle
        spine3,        # 9  spine3
        lm[31],        # 10 left_foot
        lm[32],        # 11 right_foot
        neck,          # 12 neck
        left_collar,   # 13 left_collar
        right_collar,  # 14 right_collar
        head,          # 15 head
        left_sh,       # 16 left_shoulder
        right_sh,      # 17 right_shoulder
        lm[13],        # 18 left_elbow
        lm[14],        # 19 right_elbow
        lm[15],        # 20 left_wrist
        lm[16],        # 21 right_wrist
        np.zeros(3),   # 22 orientation (placeholder)
        np.zeros(3),   # 23 translation (placeholder)
        lm[19],        # 24 left_middle2 (left index tip proxy)
        lm[20],        # 25 right_middle2 (right index tip proxy)
    ])
    return joints


def process_video(source, output_path, model_path="pose_landmarker_heavy.task"):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )
    landmarker = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(source)
    cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)

    # Setup video writer
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out_path = output_path.replace(".pt", "_annotated.mp4")
    writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    all_frames = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if cv2.getWindowProperty('MediaPipe Pose', cv2.WND_PROP_VISIBLE) < 1:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            frame_count += 1
            results = landmarker.detect_for_video(mp_image, frame_count)

            if results.pose_landmarks:
                joints = landmarks_to_motionscript_joints(results.pose_landmarks[0])
                all_frames.append(joints)
                annotated = draw_landmarks_on_image(frame_rgb, results)
            else:
                annotated = frame_rgb

            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)
            cv2.imshow('MediaPipe Pose', annotated_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        landmarker.close()

    if all_frames:
        coords = torch.tensor(np.stack(all_frames), dtype=torch.float32)
        torch.save(coords, output_path)
        print(f"Saved {coords.shape[0]} frames -> {output_path}  shape: {coords.shape}")
        print(f"Saved annotated video -> {video_out_path}")
        return coords
    else:
        print("No frames captured.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam, or path to video file")
    parser.add_argument("--output", default="my_motion.pt")
    parser.add_argument("--model", default="pose_landmarker_heavy.task")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    process_video(source=source, output_path=args.output, model_path=args.model)