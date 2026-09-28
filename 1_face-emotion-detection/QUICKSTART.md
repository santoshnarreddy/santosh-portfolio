# Quick Start — Face Emotion Detection

## Install
```bash
pip install -r requirements.txt
```

## Run (webcam)
```bash
python emotion_detector.py
```

## Run on video file
```bash
python emotion_detector.py --source path/to/video.mp4
```

## Save output
```bash
python emotion_detector.py --save
```

## Train from scratch
1. Download FER2013 from https://www.kaggle.com/datasets/msambare/fer2013
2. Unzip to `data/fer2013/` (must have `train/` and `test/` subfolders)
3. Run: `python train_model.py --data_dir ./data/fer2013`
4. Best model saved to `model/emotion_cnn.h5`

## Note
Pre-trained weights (`model/emotion_cnn.h5`) are NOT included in this ZIP
due to file size. Train using step above, or download from the GitHub Releases page.

## Controls
- `q` → quit
- `s` → save screenshot
