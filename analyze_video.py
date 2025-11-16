"""
Simple script to analyze a video directly
Place your video in the same directory and name it "videos.mp4"
The analyzed output will be saved as "analyzed.mp4"
"""

import logging
from pathlib import Path
from video_processor import VideoProcessor
from reference_loader import ReferenceLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # File paths
    input_video = Path("videos.mp4")
    reference_video = Path("nika3_perfect_reference.mp4")
    output_video = Path("analyzed.mp4")
    
    # Check if input video exists
    if not input_video.exists():
        logger.error(f"Input video not found: {input_video}")
        logger.error("Please place your video in this directory and name it 'videos.mp4'")
        return
    
    # Check if reference video exists
    if not reference_video.exists():
        logger.error(f"Reference video not found: {reference_video}")
        return
    
    logger.info("="*60)
    logger.info("KARATE FORM ANALYSIS")
    logger.info("="*60)
    
    # Load reference data
    logger.info(f"Loading reference video: {reference_video}")
    reference_loader = ReferenceLoader(reference_video)
    reference_data = reference_loader.load_reference_data()
    logger.info("✓ Reference data loaded")
    
    # Initialize video processor
    logger.info("Initializing video processor...")
    video_processor = VideoProcessor(reference_data)
    logger.info("✓ Video processor initialized")
    
    # Process video
    logger.info(f"Processing video: {input_video}")
    logger.info("This may take a few minutes...")
    
    result = video_processor.process_video(
        input_video_path=str(input_video),
        output_video_path=str(output_video)
    )
    
    if result['success']:
        logger.info("="*60)
        logger.info("✓ ANALYSIS COMPLETE!")
        logger.info("="*60)
        logger.info(f"Output video: {output_video}")
        logger.info(f"Score: {result['analysis']['score']}/100")
        logger.info(f"Grade: {result['analysis']['grade']}")
        logger.info(f"Overall Assessment: {result['analysis']['overall_assessment']}")
        logger.info("")
        logger.info("Recommendations:")
        for i, rec in enumerate(result['analysis']['recommendations'], 1):
            logger.info(f"  {i}. {rec}")
        logger.info("="*60)
    else:
        logger.error(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()
