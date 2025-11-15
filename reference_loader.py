"""
Reference Data Loader
Loads and processes Nika3 perfect reference video to extract keypoints
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# YOLO Pose keypoint indices (COCO format)
KEYPOINT_MAPPING = {
    'head': 0,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_hand': 9,
    'right_hand': 10,
    'left_hip': 11,
    'right_hip': 12,
    'chest': 5,
    'left_knee': 13,
    'right_knee': 14,
    'left_toe': 15,
    'right_toe': 16
}


class ReferenceLoader:
    """Load and process reference video (Nika3) to extract reference keypoints"""
    
    def __init__(self, reference_video_path):
        """
        Initialize reference loader
        
        Args:
            reference_video_path: Path to Nika3 perfect reference video
        """
        self.reference_video_path = Path(reference_video_path)
        self.model = YOLO('yolo11n-pose.pt')
        logger.info("✓ Reference loader initialized")
    
    def extract_keypoints_from_frame(self, frame, confidence_threshold=0.5):
        """
        Extract keypoints from a single frame using YOLOv11 pose detection
        
        Args:
            frame: Video frame (numpy array)
            confidence_threshold: Minimum confidence for keypoint detection
            
        Returns:
            Dictionary of keypoint coordinates with confidence, or None
        """
        results = self.model(frame, verbose=False, conf=0.5)
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confidences = results[0].keypoints.conf.cpu().numpy()
            
            if len(keypoints) > 0 and len(confidences) > 0:
                person_keypoints = keypoints[0]
                person_confidences = confidences[0]
                
                extracted = {}
                for name, idx in KEYPOINT_MAPPING.items():
                    x, y = person_keypoints[idx]
                    conf = person_confidences[idx]
                    
                    if x > 0 and y > 0 and conf >= confidence_threshold:
                        extracted[name] = (int(x), int(y), float(conf))
                    else:
                        extracted[name] = None
                
                # Calculate chest center from both shoulders
                left_shoulder = person_keypoints[5]
                right_shoulder = person_keypoints[6]
                left_shoulder_conf = person_confidences[5]
                right_shoulder_conf = person_confidences[6]
                
                if (left_shoulder[0] > 0 and left_shoulder[1] > 0 and left_shoulder_conf >= confidence_threshold and
                    right_shoulder[0] > 0 and right_shoulder[1] > 0 and right_shoulder_conf >= confidence_threshold):
                    chest_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
                    chest_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
                    chest_conf = (left_shoulder_conf + right_shoulder_conf) / 2
                    shoulder_width = int(abs(right_shoulder[0] - left_shoulder[0]))
                    extracted['chest'] = (chest_x, chest_y, float(chest_conf), shoulder_width)
                        
                return extracted
        
        return None
    
    def load_reference_data(self, sample_frames=30):
        """
        Load reference data from Nika3 video
        Samples frames throughout the video to capture various poses
        
        Args:
            sample_frames: Number of frames to sample from video
            
        Returns:
            Dictionary containing reference keypoints and video metadata
        """
        logger.info(f"Loading reference video: {self.reference_video_path.name}")
        
        if not self.reference_video_path.exists():
            raise FileNotFoundError(f"Reference video not found: {self.reference_video_path}")
        
        cap = cv2.VideoCapture(str(self.reference_video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open reference video: {self.reference_video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Reference video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Sample frames evenly throughout the video
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        all_keypoints = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            keypoints = self.extract_keypoints_from_frame(frame, confidence_threshold=0.6)
            if keypoints:
                all_keypoints.append(keypoints)
        
        cap.release()
        
        logger.info(f"✓ Extracted keypoints from {len(all_keypoints)} reference frames")
        
        if len(all_keypoints) == 0:
            raise ValueError("No keypoints extracted from reference video")
        
        return {
            'keypoints': all_keypoints,
            'fps': fps,
            'total_frames': total_frames,
            'resolution': (width, height),
            'reference_video': self.reference_video_path.name
        }
