"""
Video Processing Module
Handles video analysis, keypoint extraction, pose comparison, and annotation
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# YOLO Pose keypoint indices (COCO format)
KEYPOINT_MAPPING = {
    'head': 0,  # nose
    'left_elbow': 7,
    'right_elbow': 8,
    'left_hand': 9,  # wrist
    'right_hand': 10,
    'left_hip': 11,
    'right_hip': 12,
    'chest': 5,  # approximating with left shoulder
    'left_knee': 13,
    'right_knee': 14,
    'left_toe': 15,  # ankle
    'right_toe': 16
}


class VideoProcessor:
    """Process videos and compare with reference using YOLOv11 pose detection"""
    
    def __init__(self, reference_data):
        """
        Initialize video processor with reference data
        
        Args:
            reference_data: Dictionary containing reference keypoints from Nika3
        """
        self.reference_data = reference_data
        self.model = YOLO('yolo11n-pose.pt')
        logger.info("✓ YOLOv11 pose model loaded")
        
        # Body-part specific box sizes for visualization
        self.body_part_sizes = {
            'head': 25,
            'left_elbow': 18,
            'right_elbow': 18,
            'left_hand': 15,
            'right_hand': 15,
            'left_hip': 20,
            'right_hip': 20,
            'chest': 22,
            'left_knee': 18,
            'right_knee': 18,
            'left_toe': 15,
            'right_toe': 15
        }
        
        # Strict thresholds for accurate evaluation
        self.body_part_thresholds = {
            'head': 0.25,
            'left_elbow': 0.35,
            'right_elbow': 0.35,
            'left_hand': 0.4,
            'right_hand': 0.4,
            'left_hip': 0.2,
            'right_hip': 0.2,
            'chest': 0.25,
            'left_knee': 0.35,
            'right_knee': 0.35,
            'left_toe': 0.25,
            'right_toe': 0.25
        }
    
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
    
    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints relative to body size (hip width) and center position
        
        Args:
            keypoints: Dictionary of keypoint coordinates
            
        Returns:
            Normalized keypoints dictionary
        """
        if not keypoints:
            return None
        
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')
        
        if not left_hip or not right_hip:
            return keypoints
        
        hip_width = abs(left_hip[0] - right_hip[0])
        if hip_width < 10:
            hip_width = 100
        
        center_x = (left_hip[0] + right_hip[0]) / 2
        center_y = (left_hip[1] + right_hip[1]) / 2
        
        normalized = {}
        for name, data in keypoints.items():
            if data is not None:
                x, y = data[0], data[1]
                conf = data[2] if len(data) > 2 else 1.0
                
                norm_x = (x - center_x) / hip_width
                norm_y = (y - center_y) / hip_width
                
                normalized[name] = (norm_x, norm_y, conf)
            else:
                normalized[name] = None
        
        return normalized
    
    def calculate_normalized_distance(self, kp1, kp2):
        """Calculate Euclidean distance between normalized keypoints"""
        if kp1 is None or kp2 is None:
            return float('inf')
        
        x1, y1 = kp1[0], kp1[1]
        x2, y2 = kp2[0], kp2[1]
        
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def analyze_specific_deviations(self, student_kp, ref_kp, keypoint_name, distance):
        """
        Analyze specific deviations and provide meaningful feedback
        
        Args:
            student_kp: Student keypoint coordinates
            ref_kp: Reference keypoint coordinates
            keypoint_name: Name of the body part
            distance: Calculated distance between keypoints
            
        Returns:
            String describing the deviation
        """
        if student_kp is None or ref_kp is None:
            return "Missing"
        
        student_x, student_y = student_kp[0], student_kp[1]
        ref_x, ref_y = ref_kp[0], ref_kp[1]
        
        vertical_diff = student_y - ref_y
        horizontal_diff = student_x - ref_x
        
        feedback = []
        
        if 'toe' in keypoint_name:
            if abs(horizontal_diff) > 0.15:
                feedback.append("too wide" if horizontal_diff > 0 else "too narrow")
            if abs(vertical_diff) > 0.25:
                feedback.append("too low" if vertical_diff > 0 else "too high")
        else:
            if abs(vertical_diff) > 0.3:
                feedback.append("too low" if vertical_diff > 0 else "too high")
            if abs(horizontal_diff) > 0.3:
                feedback.append("too far right" if horizontal_diff > 0 else "too far left")
        
        if distance > 0.5:
            feedback.append("major deviation")
        elif distance > 0.3:
            feedback.append("moderate deviation")
        
        return ", ".join(feedback) if feedback else "minor deviation"
    
    def compare_poses_with_reference(self, student_keypoints, reference_keypoints_list):
        """
        Compare student keypoints with reference using normalized coordinates
        
        Args:
            student_keypoints: Current frame keypoints
            reference_keypoints_list: List of reference keypoints from Nika3
            
        Returns:
            Dictionary containing match results and statistics
        """
        if student_keypoints is None or not reference_keypoints_list:
            return None
        
        student_norm = self.normalize_keypoints(student_keypoints)
        if student_norm is None:
            return None
        
        keypoint_matches = {}
        match_count = 0
        total_count = 0
        
        for keypoint_name in KEYPOINT_MAPPING.keys():
            student_kp = student_norm.get(keypoint_name)
            
            if student_kp is None:
                keypoint_matches[keypoint_name] = {
                    'matched': False,
                    'distance': float('inf'),
                    'confidence': 0,
                    'deviation_type': 'Missing'
                }
                total_count += 1
                continue
            
            conf = student_kp[2] if len(student_kp) > 2 else 1.0
            
            # Find best match from reference frames
            best_distance = float('inf')
            best_ref_kp = None
            
            for ref_keypoints in reference_keypoints_list:
                ref_norm = self.normalize_keypoints(ref_keypoints)
                if ref_norm is None:
                    continue
                    
                ref_kp = ref_norm.get(keypoint_name)
                if ref_kp is None:
                    continue
                    
                distance = self.calculate_normalized_distance(student_kp, ref_kp)
                
                if distance < best_distance:
                    best_distance = distance
                    best_ref_kp = ref_kp
            
            specific_threshold = self.body_part_thresholds.get(keypoint_name, 0.4)
            matched = best_distance < specific_threshold and conf >= 0.5
            deviation_type = self.analyze_specific_deviations(student_kp, best_ref_kp, keypoint_name, best_distance)
            
            keypoint_matches[keypoint_name] = {
                'matched': matched,
                'distance': best_distance,
                'confidence': conf,
                'deviation_type': deviation_type if not matched else 'Good'
            }
            
            if matched:
                match_count += 1
            total_count += 1
        
        match_percentage = (match_count / total_count * 100) if total_count > 0 else 0
        
        return {
            'keypoint_matches': keypoint_matches,
            'match_percentage': match_percentage,
            'matched_count': match_count,
            'total_count': total_count
        }
    
    def detect_wrist_orientation(self, frame, hand_coords, elbow_coords):
        """
        Detect wrist orientation (UP/DOWN/SIDEWAYS/NEUTRAL)
        
        Args:
            frame: Video frame
            hand_coords: Hand/wrist coordinates
            elbow_coords: Elbow coordinates
            
        Returns:
            String indicating wrist position
        """
        if hand_coords is None or elbow_coords is None:
            return 'UNKNOWN'
        
        hand_y = hand_coords[1]
        elbow_y = elbow_coords[1]
        hand_x = hand_coords[0]
        elbow_x = elbow_coords[0]
        
        vertical_diff = hand_y - elbow_y
        horizontal_diff = abs(hand_x - elbow_x)
        
        vertical_threshold = 70
        horizontal_threshold = 90
        
        if vertical_diff < -vertical_threshold:
            return 'DOWN'
        elif vertical_diff > vertical_threshold:
            return 'UP'
        elif horizontal_diff > horizontal_threshold:
            return 'SIDEWAYS'
        else:
            return 'NEUTRAL'
    
    def determine_wrist_positions(self, frame, keypoints):
        """Determine wrist positions for both hands"""
        wrist_positions = {
            'left_wrist': 'UNKNOWN',
            'right_wrist': 'UNKNOWN'
        }
        
        left_hand = keypoints.get('left_hand')
        left_elbow = keypoints.get('left_elbow')
        
        if left_hand and left_elbow:
            wrist_positions['left_wrist'] = self.detect_wrist_orientation(frame, left_hand, left_elbow)
        
        right_hand = keypoints.get('right_hand')
        right_elbow = keypoints.get('right_elbow')
        
        if right_hand and right_elbow:
            wrist_positions['right_wrist'] = self.detect_wrist_orientation(frame, right_hand, right_elbow)
        
        return wrist_positions
    
    def draw_annotated_boxes(self, frame, keypoints, comparison):
        """
        Draw colored boxes on frame (GREEN for correct, RED for deviations)
        
        Args:
            frame: Video frame to annotate
            keypoints: Detected keypoints
            comparison: Comparison results from compare_poses_with_reference
            
        Returns:
            Annotated frame
        """
        wrist_pos = self.determine_wrist_positions(frame, keypoints)
        
        for keypoint_name, data in keypoints.items():
            if data is not None:
                x, y = data[0], data[1]
                
                match_info = comparison['keypoint_matches'].get(keypoint_name, {})
                
                # GREEN for matched, RED for deviations
                if match_info.get('matched', False):
                    color = (0, 255, 0)  # GREEN
                    thickness = 3
                else:
                    color = (0, 0, 255)  # RED
                    thickness = 4
                
                # Special handling for chest
                if keypoint_name == 'chest' and len(data) > 3:
                    shoulder_width = data[3]
                    half_width = shoulder_width // 2
                    half_height = 15
                    
                    cv2.rectangle(frame,
                                 (x - half_width, y - half_height),
                                 (x + half_width, y + half_height),
                                 color, thickness)
                    
                    label = 'chest'
                    if not match_info.get('matched', False):
                        deviation = match_info.get('deviation_type', 'deviation')
                        label += f" - {deviation}"
                    
                    label_y = y - half_height - 8
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                    
                    overlay = frame.copy()
                    cv2.rectangle(overlay,
                                 (x - half_width - 2, label_y - text_height - 3),
                                 (x - half_width + text_width + 2, label_y + 3),
                                 color, -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    cv2.putText(frame, label,
                               (x - half_width, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    half_size = self.body_part_sizes.get(keypoint_name, 20) // 2
                    
                    cv2.rectangle(frame,
                                 (x - half_size, y - half_size),
                                 (x + half_size, y + half_size),
                                 color, thickness)
                    
                    label = keypoint_name.replace('_', ' ')
                    
                    if keypoint_name == 'left_hand':
                        label += f" [{wrist_pos['left_wrist']}]"
                    elif keypoint_name == 'right_hand':
                        label += f" [{wrist_pos['right_wrist']}]"
                    
                    if not match_info.get('matched', False):
                        deviation = match_info.get('deviation_type', 'deviation')
                        label += f" - {deviation}"
                    
                    label_y = y - half_size - 8
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                    
                    overlay = frame.copy()
                    cv2.rectangle(overlay,
                                 (x - half_size - 2, label_y - text_height - 3),
                                 (x - half_size + text_width + 2, label_y + 3),
                                 color, -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    cv2.putText(frame, label,
                               (x - half_size, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def add_info_overlay(self, frame, cumulative_avg, frame_count, total_frames, wrist_pos):
        """
        Add information overlay to frame (score, frame count, wrist positions)
        
        Args:
            frame: Video frame
            cumulative_avg: Running average score
            frame_count: Current frame number
            total_frames: Total frames in video
            wrist_pos: Wrist position dictionary
            
        Returns:
            Frame with overlay
        """
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Color code the score
        if cumulative_avg >= 70:
            perf_color = (0, 255, 0)  # GREEN
        elif cumulative_avg >= 50:
            perf_color = (0, 255, 255)  # YELLOW
        else:
            perf_color = (0, 0, 255)  # RED
        
        cv2.putText(frame, f"Avg Score: {cumulative_avg:.1f}%",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, perf_color, 2, cv2.LINE_AA)
        
        wrist_info = f"Wrists: L-{wrist_pos['left_wrist']} | R-{wrist_pos['right_wrist']}"
        cv2.putText(frame, wrist_info,
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def generate_analysis_report(self, frame_results, total_frames):
        """
        Generate comprehensive JSON analysis report
        
        Args:
            frame_results: List of frame analysis results
            total_frames: Total frames processed
            
        Returns:
            Dictionary containing comprehensive analysis
        """
        if not frame_results:
            return {
                'error': 'No frames analyzed',
                'score': 0,
                'status': 'failed'
            }
        
        # Calculate overall statistics
        avg_match_percentage = np.mean([r['match_percentage'] for r in frame_results])
        score = int(avg_match_percentage)
        
        # Aggregate keypoint statistics
        keypoint_stats = {kp: {
            'matched': 0,
            'total': 0,
            'avg_distance': [],
            'deviation_types': []
        } for kp in KEYPOINT_MAPPING.keys()}
        
        for result in frame_results:
            for kp_name, match_info in result['comparison']['keypoint_matches'].items():
                keypoint_stats[kp_name]['total'] += 1
                if match_info['matched']:
                    keypoint_stats[kp_name]['matched'] += 1
                if match_info['distance'] != float('inf'):
                    keypoint_stats[kp_name]['avg_distance'].append(match_info['distance'])
                if not match_info['matched'] and match_info.get('deviation_type'):
                    keypoint_stats[kp_name]['deviation_types'].append(match_info['deviation_type'])
        
        # Calculate body part scores
        body_part_analysis = {}
        poor_parts = []
        
        for kp_name, stats in keypoint_stats.items():
            if stats['total'] > 0:
                match_rate = (stats['matched'] / stats['total']) * 100
                
                status = "excellent" if match_rate >= 80 else \
                        "good" if match_rate >= 60 else \
                        "needs_work" if match_rate >= 40 else "poor"
                
                # Get most common deviation
                common_issue = "None"
                if stats['deviation_types']:
                    from collections import Counter
                    deviation_counts = Counter(stats['deviation_types'])
                    most_common = deviation_counts.most_common(1)
                    if most_common:
                        common_issue = most_common[0][0]
                
                body_part_analysis[kp_name] = {
                    'score': round(match_rate, 1),
                    'status': status,
                    'common_issue': common_issue,
                    'frames_analyzed': stats['total'],
                    'frames_correct': stats['matched']
                }
                
                if match_rate < 60:
                    poor_parts.append((kp_name, match_rate, stats))
        
        # Generate recommendations
        recommendations = []
        poor_parts.sort(key=lambda x: x[1])
        
        for kp_name, match_rate, stats in poor_parts[:5]:
            if stats['deviation_types']:
                from collections import Counter
                deviation_counts = Counter(stats['deviation_types'])
                most_common = deviation_counts.most_common(1)[0]
                
                body_part = kp_name.replace('_', ' ').title()
                
                if 'too low' in most_common[0]:
                    recommendations.append(f"Raise your {body_part} higher")
                elif 'too high' in most_common[0]:
                    recommendations.append(f"Lower your {body_part}")
                elif 'too far left' in most_common[0]:
                    recommendations.append(f"Move your {body_part} more to the right")
                elif 'too far right' in most_common[0]:
                    recommendations.append(f"Move your {body_part} more to the left")
                elif 'major deviation' in most_common[0]:
                    recommendations.append(f"Focus on {body_part} positioning - significant adjustment needed")
        
        # Performance assessment
        if score >= 90:
            assessment = "EXCELLENT! Form closely matches the perfect reference."
            grade_label = "A+"
        elif score >= 75:
            assessment = "GOOD! Form is solid with some minor deviations."
            grade_label = "B"
        elif score >= 60:
            assessment = "FAIR. Form shows promise but needs improvement."
            grade_label = "C"
        elif score >= 40:
            assessment = "NEEDS WORK. Several areas need attention."
            grade_label = "D"
        else:
            assessment = "BEGINNER. Focus on mastering the basic form."
            grade_label = "F"
        
        # Aggregate wrist positions
        left_wrist_positions = {'UP': 0, 'DOWN': 0, 'SIDEWAYS': 0, 'NEUTRAL': 0, 'UNKNOWN': 0}
        right_wrist_positions = {'UP': 0, 'DOWN': 0, 'SIDEWAYS': 0, 'NEUTRAL': 0, 'UNKNOWN': 0}
        
        for result in frame_results:
            if 'wrist_positions' in result:
                wrist_pos = result['wrist_positions']
                left_wrist_positions[wrist_pos['left_wrist']] += 1
                right_wrist_positions[wrist_pos['right_wrist']] += 1
        
        # Build final report
        report = {
            'score': score,
            'grade': grade_label,
            'assessment': assessment,
            'statistics': {
                'total_frames_analyzed': len(frame_results),
                'average_match_percentage': round(avg_match_percentage, 2),
                'reference_video': 'nika3_perfect_reference.mp4'
            },
            'body_part_analysis': body_part_analysis,
            'recommendations': recommendations,
            'wrist_analysis': {
                'left_wrist': {k: round(v / len(frame_results) * 100, 1) for k, v in left_wrist_positions.items() if v > 0},
                'right_wrist': {k: round(v / len(frame_results) * 100, 1) for k, v in right_wrist_positions.items() if v > 0}
            }
        }
        
        return report
    
    def process_video(self, input_video_path, output_video_path, output_json_path):
        """
        Main video processing function
        
        Args:
            input_video_path: Path to uploaded video
            output_video_path: Path for annotated output video
            output_json_path: Path for JSON analysis report
            
        Returns:
            Analysis report dictionary
        """
        logger.info(f"Processing video: {input_video_path}")
        
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        cumulative_match_sum = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Extract keypoints
            student_keypoints = self.extract_keypoints_from_frame(frame, confidence_threshold=0.6)
            
            if student_keypoints:
                # Compare with reference
                comparison = self.compare_poses_with_reference(
                    student_keypoints,
                    self.reference_data['keypoints']
                )
                
                if comparison:
                    # Draw annotated boxes
                    frame = self.draw_annotated_boxes(frame, student_keypoints, comparison)
                    
                    # Calculate cumulative average
                    match_pct = comparison['match_percentage']
                    cumulative_match_sum += match_pct
                    cumulative_avg = cumulative_match_sum / len(frame_results) if frame_results else match_pct
                    
                    # Get wrist positions
                    wrist_pos = self.determine_wrist_positions(frame, student_keypoints)
                    
                    # Add info overlay
                    frame = self.add_info_overlay(frame, cumulative_avg, frame_count, total_frames, wrist_pos)
                    
                    # Store results
                    frame_results.append({
                        'frame': frame_count,
                        'match_percentage': match_pct,
                        'comparison': comparison,
                        'wrist_positions': wrist_pos
                    })
            
            # Write frame
            out.write(frame)
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        logger.info(f"✓ Video processing complete: {output_video_path}")
        
        # Generate analysis report
        analysis_report = self.generate_analysis_report(frame_results, total_frames)
        
        # Save JSON report
        with open(output_json_path, 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        logger.info(f"✓ Analysis report saved: {output_json_path}")
        
        return analysis_report
