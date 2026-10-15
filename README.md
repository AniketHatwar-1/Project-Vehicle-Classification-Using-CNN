# Vehicle Classification CNN

A simple Windows project using a custom CNN trained from random initialization. No pretrained YOLO or pretrained classification model is used.

## Dataset
This project expects `Dataset1.zip` beside `train.py`. The ZIP should contain class folders for `Bus`, `Car`, `Truck`, and `motorcycle`, with images inside. The script extracts valid images into `data/images/` automatically.

## Run
1. Extract this project to a short path such as `C:\\VehicleCNN`.
2. Copy your existing `Dataset1.zip` into the project folder (beside `train.py`).
3. If you already have a working `.venv` from the previous project, it can usually be reused. Otherwise double-click `setup.bat`.
4. Double-click `train.bat`; it trains up to 30 epochs and may stop early.
5. Double-click `app.bat` to open the upload interface.

## Results
- `results/training_history.csv`
- `results/test_results.txt`
- `results/accuracy_curve.png`
- `results/loss_curve.png`
- `models/vehicle_cnn.pth`

## Important
Training augmentation is applied only to training images. Validation and test images remain unchanged. The data split is deterministic (70% train, 15% validation, 15% test). Do not use the test set to choose model settings. With 100 images per class, results are a baseline and can vary. More diverse, correctly labelled images are the most useful next improvement.
