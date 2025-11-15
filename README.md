# Karate Form Analysis API

AI-powered REST API for analyzing karate form using YOLOv11 pose detection. Compares user videos against professional reference (Nika3) and provides detailed performance analysis with annotated videos.

## 🚀 Features

- **Video Upload & Analysis**: Upload karate form videos via REST API
- **Real-time Pose Detection**: YOLOv11-based keypoint detection
- **Smart Comparison**: Normalized comparison against Nika3 perfect reference
- **Visual Feedback**: Green boxes for correct form, red for deviations
- **Comprehensive Analysis**: JSON report with score (1-100), body part breakdown, recommendations
- **Railway Deployment**: Ready to deploy on Railway.app

## 📋 API Endpoints

### 1. Health Check
```
GET /health
```
Check if API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "reference_loaded": true
}
```

### 2. Analyze Video
```
POST /api/analyze
Content-Type: multipart/form-data
```
Upload a video for analysis.

**Request:**
- `video`: Video file (mp4, avi, mov, mkv, webm) - Max 100MB

**Response:**
```json
{
  "video_id": "uuid-string",
  "status": "completed",
  "analysis": {
    "score": 85,
    "grade": "B",
    "assessment": "GOOD! Form is solid with some minor deviations.",
    "statistics": {
      "total_frames_analyzed": 120,
      "average_match_percentage": 85.2,
      "reference_video": "nika3_perfect_reference.mp4"
    },
    "body_part_analysis": {
      "left_hand": {
        "score": 78.5,
        "status": "good",
        "common_issue": "too low",
        "frames_analyzed": 120,
        "frames_correct": 94
      }
    },
    "recommendations": [
      "Raise your Left Hand higher",
      "Move your Right Knee more to the left"
    ],
    "wrist_analysis": {
      "left_wrist": {"DOWN": 85.2, "UP": 14.8},
      "right_wrist": {"SIDEWAYS": 100.0}
    }
  },
  "video_url": "/api/video/{video_id}",
  "json_url": "/api/analysis/{video_id}"
}
```

### 3. Get Analyzed Video
```
GET /api/video/{video_id}
```
Download the processed video with green/red annotations.

**Response:** Video file (mp4)

### 4. Get Analysis JSON
```
GET /api/analysis/{video_id}
```
Download the complete JSON analysis report.

**Response:** JSON file

### 5. Cleanup Files
```
DELETE /api/cleanup/{video_id}
```
Remove processed files for a specific video.

**Response:**
```json
{
  "status": "success",
  "deleted": ["video", "json"],
  "message": "Cleaned up 2 file(s)"
}
```

## 🛠️ Local Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MuayML
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure reference video exists:
   - Place `nika3_perfect_reference.mp4` in the project root
   - This is the perfect form reference video used for comparison

5. Run the application:
```bash
python app.py
```

The API will start on `http://localhost:5000`

## 🚂 Railway Deployment

### Prerequisites
- Railway account (https://railway.app)
- Git repository

### Deployment Steps

1. **Prepare Repository**
   - Ensure `nika3_perfect_reference.mp4` is committed to the repository
   - Verify all files are present:
     - `app.py`
     - `video_processor.py`
     - `reference_loader.py`
     - `requirements.txt`
     - `Procfile`
     - `railway.json`
     - `nika3_perfect_reference.mp4`

2. **Deploy to Railway**

   **Option A: Railway CLI**
   ```bash
   npm i -g @railway/cli
   railway login
   railway init
   railway up
   ```

   **Option B: GitHub Integration**
   - Connect your GitHub repository to Railway
   - Railway will auto-detect the configuration
   - Deploy automatically on push

3. **Configure Environment**
   - Railway automatically sets `PORT` environment variable
   - No additional configuration needed

4. **Monitor Deployment**
   - Check build logs in Railway dashboard
   - Wait for "Model initialization complete!" message
   - Test with `/health` endpoint

### Important Notes

- **Model Loading**: First startup takes 1-2 minutes (downloads YOLOv11 model)
- **Memory**: Requires at least 1GB RAM (Railway Hobby plan sufficient)
- **Timeout**: Video processing timeout set to 300 seconds
- **Workers**: Single worker with 2 threads for optimal performance
- **Storage**: Uploads/outputs stored in ephemeral storage (auto-cleanup recommended)

## 📊 Analysis Output

### Video Annotations
- **Green Boxes**: Body part position matches reference (correct form)
- **Red Boxes**: Deviation from reference (needs correction)
- **Labels**: Body part name + position info + deviation type
- **Overlay**: Running average score, frame counter, wrist positions

### JSON Report Structure
```json
{
  "score": 85,
  "grade": "B",
  "assessment": "Performance description",
  "statistics": {
    "total_frames_analyzed": 120,
    "average_match_percentage": 85.2,
    "reference_video": "nika3_perfect_reference.mp4"
  },
  "body_part_analysis": {
    "body_part_name": {
      "score": 78.5,
      "status": "good|excellent|needs_work|poor",
      "common_issue": "Description of most common deviation",
      "frames_analyzed": 120,
      "frames_correct": 94
    }
  },
  "recommendations": [
    "Specific actionable feedback"
  ],
  "wrist_analysis": {
    "left_wrist": {"position": percentage},
    "right_wrist": {"position": percentage}
  }
}
```

## 🔧 Configuration

### Adjust Thresholds
Modify thresholds in `video_processor.py`:
```python
self.body_part_thresholds = {
    'head': 0.25,
    'left_hand': 0.4,
    # adjust values for stricter/looser comparison
}
```

## 📁 Project Structure

```
MuayML/
├── app.py                          # Flask REST API
├── video_processor.py              # Video analysis logic
├── reference_loader.py             # Reference data loader
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway process configuration
├── railway.json                    # Railway deployment config
├── .env.example                    # Environment template
├── nika3_perfect_reference.mp4     # Perfect form reference video
├── karate.ipynb                    # Original notebook (reference)
├── uploads/                        # Temporary upload storage
└── outputs/                        # Processed videos & JSON
```

## 🧪 Testing

### Test with cURL
```bash
# Health check
curl http://localhost:5000/health

# Upload video
curl -X POST http://localhost:5000/api/analyze -F "video=@test_video.mp4"

# Download analyzed video
curl http://localhost:5000/api/video/{video_id} -o analyzed.mp4

# Download JSON analysis
curl http://localhost:5000/api/analysis/{video_id} -o analysis.json
```

### Test with Python
```python
import requests

# Upload video
with open('test_video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/analyze',
        files={'video': f}
    )
    result = response.json()
    video_id = result['video_id']

# Download analyzed video
video_response = requests.get(f'http://localhost:5000/api/video/{video_id}')
with open('analyzed.mp4', 'wb') as f:
    f.write(video_response.content)

# Get JSON analysis
json_response = requests.get(f'http://localhost:5000/api/analysis/{video_id}')
analysis = json_response.json()
print(f"Score: {analysis['score']}/100")
```

## 🐛 Troubleshooting

### "Reference video not found"
- Ensure `nika3_perfect_reference.mp4` is in the project root
- Check file name matches exactly (case-sensitive)

### "Model loading timeout"
- First run downloads YOLOv11 model (~6MB)
- Subsequent runs use cached model
- Increase timeout in `railway.json` if needed

### "File too large" error
- Maximum upload size: 100MB
- Adjust in `app.py`: `app.config['MAX_CONTENT_LENGTH']`

### Out of memory
- Reduce video resolution before upload
- Use Railway Pro plan for more RAM
- Adjust workers/threads in `Procfile`