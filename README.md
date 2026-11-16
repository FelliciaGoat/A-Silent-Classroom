# A-Silent-Classroom
AI system that flags students who may have a doubt in classroom photos, videos and webcam streams, using raised-hand, pointing-finger, attention and expression detection. Flask + ngrok backend on Google Colab with a web UI.
# Silent Classroom

AI-based student doubt detection. In a large or online class, a teacher can't see who is confused but too shy to ask. Silent Classroom analyses a classroom **photo, video or live webcam stream** and draws one annotated frame with boxes, emotions and a **DOUBT ALERT** for students who may have a doubt.

## What it detects
- **Raised hands:** YOLO11s-pose (wrist above shoulder)
- **Pointing index finger / open palm:** MediaPipe hand landmarks
- **Attention:** MediaPipe FaceLandmarker (head direction: looking at the front or away)
- **Emotion:** DeepFace, with a rough confusion score (surprise + fear + sad)

A student is flagged when an arm is raised, an index finger is pointing up, or the confusion score is above 0.45. Frame skipping (gestures every 2nd frame, emotions every 10th) keeps the frame rate usable.

## Project structure
| File | Purpose |
|---|---|
| `main_pipeline.py` | The three detectors in one frame-by-frame loop (also runs from the command line) |
| `colab_backend.py` | Flask API (`/predict`, `/get_eda`, `/health`) exposed through ngrok |
| `index.html` | Web UI: Photo, Live webcam, Video, EDA graphs, Colab notebook tabs |
| `train_emotion_cnn.py` | Optional: trains an emotion CNN on FER2013 |
| `Silent_Classroom_Project_Report.docx` | Project report |

## Quick start (Google Colab)
1. Open the notebook: https://colab.research.google.com/drive/1ECh1OZ_7dmj78yfV2suaMqe7sLJmj6G?usp=sharing
2. Use **Runtime → Change runtime type → T4 GPU**.
3. Upload `main_pipeline.py` and `index.html` to `/content`.
4. Paste `colab_backend.py` into a cell and set `NGROK_TOKEN` to your own token from https://dashboard.ngrok.com. **Never commit your token.**
5. Run the cell and open the `https://….ngrok-free.dev` link it prints. The page connects to the backend automatically. The webcam works only over this https link.

## Run locally
```bash
pip install ultralytics mediapipe deepface tf-keras opencv-python
python main_pipeline.py --source 0 --show                     # webcam
python main_pipeline.py --source class.mp4 --output out.mp4   # video
python main_pipeline.py --image photo.jpg --output out.jpg    # single image
```

## Dataset and results
- **Dataset:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013), 35,887 face images, 7 emotions.
- **Baseline models** on a 1,200-image train and 400-image test subset (24×24 grayscale pixels):

| Model | Test accuracy |
|---|---|
| Logistic Regression | 32.50% |
| Random Forest | 32.50% |
| **Support Vector Machine** | **34.25%** |
| Decision Tree | 22.75% |

These are classical baselines. The live system uses pre-trained deep models, and its overall accuracy has **not** been measured.

## Limitations
- The confusion score is a rough rule on emotion probabilities, not validated confusion detection. Raised hands and pointing fingers are more reliable.
- No validation on labelled real-classroom data yet.
- Small or distant faces reduce detection quality.
- The tool processes faces: use it with consent and don't store frames.

## Future work
- Train a confusion classifier on a dataset with confusion labels (e.g. DAiSEE).
- Fine-tune a CNN on the full FER2013 dataset.
- Host the app permanently instead of using ngrok.

## Team
Naresh ([UID]) · [Teammate name (UID)] · Chandigarh University, [Branch], [Supervisor name]

## Acknowledgements
FER2013 (Kaggle), MediaPipe, Ultralytics YOLO11, DeepFace, scikit-learn, Flask, ngrok.
