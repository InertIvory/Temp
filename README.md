# Corrugator Vision

A Streamlit + OpenCV prototype for investigating computer vision on corrugated board.

## Version 1: yellow starter-marker detection

- Upload a still image of corrugated board.
- Segment pale-yellow masking tape in HSV colour space.
- Tune hue, saturation, brightness, minimum area, maximum candidate area, and morphological cleanup.
- Inspect candidate contours, the largest candidate, its bounding box, and its image centre.
- View grayscale and Canny edge outputs in the image-processing lab.

**Important:** Version 1 detects yellow-region candidates only. The crosshair marks the centre of the largest candidate, not the starter point. Version 2 will detect the board edge, estimate the tape centreline, and calculate their intersection.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

The app is designed for Streamlit Community Cloud. Commit changes to the connected GitHub branch to trigger a redeployment.

## Safety and scope

This is an experimental still-image analysis prototype. It has no live camera, PLC, or actuator connection. Do not use its output to trigger or control machinery. Any future industrial integration requires representative validation, fail-safe design, and an appropriate machine risk assessment.
