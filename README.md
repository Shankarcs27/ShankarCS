# Yoga Pose Estimation

## Overview
Yoga Pose Estimation is an AI-powered web application that helps users perfect their yoga practice by providing real-time pose analysis and personalized feedback. The system uses computer vision and machine learning to analyze body posture from images or webcam, compare it to reference poses, and guide users with expert instructions.

## Features
- **Real-time Pose Analysis:** Get instant feedback on your yoga pose alignment using advanced AI technology.
- **Progress Tracking:** Monitor your improvement with detailed progress reports and statistics.
- **Expert Guidance:** Access detailed instructions and tips for each yoga pose.
- **Voice Instructions:** Listen to step-by-step guidance for each pose.
- **Interactive Web Interface:** User-friendly interface with pose analyzer and reference images.

## Yoga Poses Supported
- Pranamasana (Prayer Pose)
- Hastauttanasana (Raised Arms Pose)
- Padangusthasana (Hand to Foot Pose)
- Ashwa Sanchalanasana (Equestrian Pose)
- Chaturanga Dandasana (Four-Limbed Staff Pose)
- Ashtanga Namaskara (Eight-Limbed Salutation)
- Bhujangasana (Cobra Pose)
- Adho Mukha Svanasana (Downward-Facing Dog)

## Project Structure
```
Yogaa Pose Estimation/
├── app.py                # Main Flask application
├── model/                # Model training, prediction, and datasets
│   ├── angles.py
│   ├── predit.py
│   ├── main.py
│   ├── angles.xlsx
│   ├── output_angles.xlsx
│   ├── pose_to_angles_predictor.pkl
│   └── DATASET/         # Images for each pose
├── static/               # CSS, JS, and static assets
├── templates/            # HTML templates
└── README.md
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd "Yogaa Pose Estimation"
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   Create a `requirements.txt` with the following content:
   ```
   flask
   opencv-python
   mediapipe
   numpy
   joblib
   openpyxl
   scikit-learn
   pandas
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare the Dataset:**
   - Place your yoga pose images in the `model/DATASET/` directory, organized by pose name.
2. **Train the Model (if needed):**
   - Run `model/predit.py` to train and save the pose-to-angles predictor model.
3. **Start the Web Application:**
   ```bash
   python app.py
   ```
   - The app will be available at `http://localhost:5000`.
4. **Navigate the Interface:**
   - Use the web interface to select a pose, start the analyzer, and receive feedback and instructions.

## Dependencies
- Flask
- OpenCV (opencv-python)
- MediaPipe
- NumPy
- Joblib
- Openpyxl
- scikit-learn
- pandas

## License
This project is for educational and personal use. Please check individual file headers for more information.

## Acknowledgements
- Yoga pose images and instructions are for demonstration and educational purposes.
- Built with [Flask](https://flask.palletsprojects.com/), [MediaPipe](https://mediapipe.dev/), and [OpenCV](https://opencv.org/). 