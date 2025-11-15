"""
Flask REST API for Karate Form Analysis
Receives video uploads, analyzes form against Nika3 reference, returns annotated video + JSON analysis
"""

import os
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from pathlib import Path

from video_processor import VideoProcessor
from reference_loader import ReferenceLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['OUTPUT_FOLDER'] = Path('outputs')
app.config['REFERENCE_VIDEO'] = Path('nika3_perfect_reference.mp4')

# Create necessary directories
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Global reference data and processor (loaded once at startup)
reference_loader = None
video_processor = None


def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_model():
    """Initialize reference data and video processor at startup"""
    global reference_loader, video_processor
    
    logger.info("Initializing Karate Form Analysis Model...")
    
    # Check if reference video exists
    if not app.config['REFERENCE_VIDEO'].exists():
        logger.error(f"Reference video not found: {app.config['REFERENCE_VIDEO']}")
        raise FileNotFoundError(f"Reference video not found: {app.config['REFERENCE_VIDEO']}")
    
    # Load reference data from Nika3 video
    logger.info("Loading Nika3 perfect reference data...")
    reference_loader = ReferenceLoader(app.config['REFERENCE_VIDEO'])
    reference_data = reference_loader.load_reference_data()
    
    # Initialize video processor
    logger.info("Initializing video processor...")
    video_processor = VideoProcessor(reference_data)
    
    logger.info("✓ Model initialization complete!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': video_processor is not None,
        'reference_loaded': reference_loader is not None
    }), 200


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """
    Main endpoint to analyze video
    
    Expected: 
    - multipart/form-data with 'video' file, OR
    - JSON with 'video_url' field containing GCS/Cloud URL
    
    Returns: JSON with video_id and analysis results
    """
    try:
        video_id = str(uuid.uuid4())
        input_path = None
        
        # Check if video URL is provided (JSON request)
        if request.is_json or request.content_type == 'application/json':
            data = request.get_json()
            video_url = data.get('video_url')
            
            if not video_url:
                return jsonify({'error': 'No video_url provided in JSON'}), 400
            
            logger.info(f"Receiving video URL: {video_url} (ID: {video_id})")
            
            # Download video from URL
            import requests as req
            try:
                video_response = req.get(video_url, timeout=120)
                video_response.raise_for_status()
                
                # Detect file extension from URL or content-type
                file_ext = 'mp4'  # default
                if '.' in video_url.split('/')[-1]:
                    file_ext = video_url.split('.')[-1].lower()
                    if file_ext not in ALLOWED_EXTENSIONS:
                        file_ext = 'mp4'
                
                input_path = app.config['UPLOAD_FOLDER'] / f"{video_id}_input.{file_ext}"
                
                # Save downloaded video
                with open(input_path, 'wb') as f:
                    f.write(video_response.content)
                
                logger.info(f"✓ Video downloaded from URL ({len(video_response.content)} bytes)")
                
            except req.exceptions.RequestException as e:
                logger.error(f"Failed to download video from URL: {str(e)}")
                return jsonify({'error': f'Failed to download video: {str(e)}'}), 400
        
        # Check if video file is uploaded (multipart/form-data)
        elif 'video' in request.files:
            file = request.files['video']
            
            # Check if filename is empty
            if file.filename == '':
                return jsonify({'error': 'No video file selected'}), 400
            
            # Validate file extension
            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400
            
            # Secure filename and save upload
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            input_path = app.config['UPLOAD_FOLDER'] / f"{video_id}_input.{file_ext}"
            
            logger.info(f"Receiving video upload: {filename} (ID: {video_id})")
            file.save(str(input_path))
        
        else:
            return jsonify({
                'error': 'No video provided. Send either: 1) multipart/form-data with "video" file, or 2) JSON with "video_url" field'
            }), 400
        
        # Define output paths
        output_video_path = app.config['OUTPUT_FOLDER'] / f"{video_id}_analyzed.mp4"
        output_json_path = app.config['OUTPUT_FOLDER'] / f"{video_id}_analysis.json"
        
        # Process video
        logger.info(f"Processing video {video_id}...")
        analysis_result = video_processor.process_video(
            input_video_path=input_path,
            output_video_path=output_video_path,
            output_json_path=output_json_path
        )
        
        # Clean up input file
        if input_path and input_path.exists():
            input_path.unlink()
        
        logger.info(f"✓ Video {video_id} processed successfully!")
        
        # Check if client wants video buffer in response
        include_video = request.args.get('include_video', 'false').lower() == 'true'
        
        response_data = {
            'video_id': video_id,
            'status': 'completed',
            'analysis': analysis_result,
            'video_url': f"/api/video/{video_id}",
            'json_url': f"/api/analysis/{video_id}"
        }
        
        # Optionally include video as base64 in response
        if include_video:
            import base64
            with open(output_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                response_data['video_data'] = {
                    'base64': video_base64,
                    'size_bytes': len(video_bytes),
                    'mime_type': 'video/mp4'
                }
            logger.info(f"✓ Included video buffer in response ({len(video_bytes)} bytes)")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/video/<video_id>', methods=['GET'])
def get_analyzed_video(video_id):
    """
    Retrieve processed video with annotations
    
    Args:
        video_id: UUID of the processed video
        format: Optional query param - 'file' (default) or 'base64'
    
    Returns: Video file (mp4) or JSON with base64 data
    """
    try:
        video_path = app.config['OUTPUT_FOLDER'] / f"{video_id}_analyzed.mp4"
        
        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if base64 format is requested
        format_type = request.args.get('format', 'file').lower()
        
        if format_type == 'base64':
            # Return as base64 in JSON for easy backend integration
            import base64
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            return jsonify({
                'video_id': video_id,
                'video_data': video_base64,
                'size_bytes': len(video_bytes),
                'mime_type': 'video/mp4',
                'filename': f'analyzed_{video_id}.mp4'
            }), 200
        
        elif format_type == 'buffer' or format_type == 'bytes':
            # Return raw bytes
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            
            from flask import Response
            return Response(
                video_bytes,
                mimetype='video/mp4',
                headers={
                    'Content-Disposition': f'inline; filename=analyzed_{video_id}.mp4',
                    'Content-Length': str(len(video_bytes))
                }
            )
        
        else:
            # Return as downloadable file (default)
            return send_file(
                str(video_path),
                mimetype='video/mp4',
                as_attachment=True,
                download_name=f"analyzed_{video_id}.mp4"
            )
        
    except Exception as e:
        logger.error(f"Error retrieving video: {str(e)}")
        return jsonify({'error': f'Failed to retrieve video: {str(e)}'}), 500


@app.route('/api/analysis/<video_id>', methods=['GET'])
def get_analysis_json(video_id):
    """
    Retrieve JSON analysis report
    
    Args:
        video_id: UUID of the processed video
    
    Returns: JSON analysis file
    """
    try:
        json_path = app.config['OUTPUT_FOLDER'] / f"{video_id}_analysis.json"
        
        if not json_path.exists():
            return jsonify({'error': 'Analysis not found'}), 404
        
        return send_file(
            str(json_path),
            mimetype='application/json',
            as_attachment=True,
            download_name=f"analysis_{video_id}.json"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        return jsonify({'error': f'Failed to retrieve analysis: {str(e)}'}), 500


@app.route('/api/cleanup/<video_id>', methods=['DELETE'])
def cleanup_files(video_id):
    """
    Clean up processed files for a given video_id
    
    Args:
        video_id: UUID of the video to clean up
    
    Returns: Success message
    """
    try:
        video_path = app.config['OUTPUT_FOLDER'] / f"{video_id}_analyzed.mp4"
        json_path = app.config['OUTPUT_FOLDER'] / f"{video_id}_analysis.json"
        
        deleted = []
        
        if video_path.exists():
            video_path.unlink()
            deleted.append('video')
        
        if json_path.exists():
            json_path.unlink()
            deleted.append('json')
        
        if deleted:
            return jsonify({
                'status': 'success',
                'deleted': deleted,
                'message': f'Cleaned up {len(deleted)} file(s)'
            }), 200
        else:
            return jsonify({'error': 'No files found to delete'}), 404
            
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 100MB'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize model before starting server
    initialize_model()
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
