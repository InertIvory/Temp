import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Corrugator Vision", page_icon="⚙️", layout="wide")

st.title("⚙️ Corrugator Vision")
st.caption("Computer-vision workbench · Version 1: yellow starter-marker detection")

st.info(
    "This version looks for the pale-yellow masking tape in a still image. "
    "It marks candidate tape regions and reports the centre of the largest candidate. "
    "It does NOT yet calculate the starter point at the board-edge intersection."
)

with st.sidebar:
    st.header("Image input")
    uploaded = st.file_uploader("Upload a board image", type=["png", "jpg", "jpeg", "bmp"])
    st.divider()
    st.header("Yellow marker tuning")
    st.caption("Adjust the thresholds if the tape is missed or other areas are selected.")
    hue_min, hue_max = st.slider("Hue range", 0, 179, (12, 45))
    sat_min = st.slider("Minimum saturation", 0, 255, 20)
    val_min = st.slider("Minimum brightness", 0, 255, 145)
    min_area = st.slider("Minimum candidate area (pixels)", 50, 30000, 700, 50)
    max_area_percent = st.slider("Maximum candidate area (% of image)", 1, 60, 25)
    kernel_size = st.select_slider("Noise-removal strength", options=[1, 3, 5, 7, 9], value=5)
    show_mask = st.checkbox("Show yellow segmentation mask", value=True)

if uploaded is None:
    st.markdown("### Upload a starter-point photograph")
    st.write(
        "Use a clear image where the yellow tape and the corrugated-board edge are visible. "
        "The first test is colour-based marker detection; edge-intersection detection comes next."
    )
    st.warning("No image loaded. Choose a photograph using the uploader in the sidebar.")
    st.stop()

try:
    rgb = np.array(Image.open(uploaded).convert("RGB"))
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

# Convert RGB input to OpenCV HSV (OpenCV hue range: 0–179).
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
lower = np.array([hue_min, sat_min, val_min], dtype=np.uint8)
upper = np.array([hue_max, 255, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower, upper)

# Morphological opening removes isolated noise; closing fills small gaps.
if kernel_size > 1:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_area = rgb.shape[0] * rgb.shape[1]
max_area = image_area * (max_area_percent / 100.0)

candidates = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        continue
    rotated = cv2.minAreaRect(contour)
    candidates.append({
        "contour": contour,
        "area": area,
        "box": (x, y, w, h),
        "rotated_box": cv2.boxPoints(rotated).astype(np.int32),
        "centre": (int(rotated[0][0]), int(rotated[0][1])),
    })

candidates.sort(key=lambda item: item["area"], reverse=True)
annotated = rgb.copy()

for index, candidate in enumerate(candidates):
    x, y, w, h = candidate["box"]
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
    cv2.putText(
        annotated, f"Candidate {index + 1}",
        (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 220), 2
    )

if candidates:
    best = candidates[0]
    x, y, w, h = best["box"]
    cx, cy = best["centre"]
    cv2.drawContours(annotated, [best["rotated_box"]], 0, (0, 255, 0), 3)
    cv2.drawMarker(annotated, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 28, 3)
    cv2.putText(
        annotated, "LARGEST CANDIDATE",
        (max(8, x), max(28, y - 30)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 130, 0), 2
    )

tab_detect, tab_lab, tab_notes = st.tabs(
    ["🎯 Starter marker detection", "🧪 Image-processing lab", "📋 Project notes"]
)

with tab_detect:
    st.subheader("Marker detection result")
    col1, col2 = st.columns(2)
    col1.image(rgb, caption="Original image", use_container_width=True)
    col2.image(annotated, caption="Candidate regions (green = largest candidate)", use_container_width=True)

    if show_mask:
        st.image(mask, caption="Yellow segmentation mask", clamp=True, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Image dimensions", f"{rgb.shape[1]} × {rgb.shape[0]} px")
    m2.metric("Candidates", len(candidates))
    m3.metric("Detection status", "CANDIDATE FOUND" if candidates else "NOT FOUND")

    if candidates:
        best = candidates[0]
        x, y, w, h = best["box"]
        cx, cy = best["centre"]
        st.success("A yellow-region candidate was found. Check the overlay to confirm it is the masking tape.")
        st.write({
            "Largest candidate centre (image pixels)": {"x": cx, "y": cy},
            "Bounding box": {"x": x, "y": y, "width": w, "height": h},
            "Contour area (pixels²)": round(best["area"], 1),
        })
    else:
        st.warning(
            "No candidate passed the current filters. Try lowering minimum brightness or saturation, "
            "reducing minimum area, widening the hue range, or increasing maximum candidate area."
        )

    st.caption(
        "The crosshair marks the detected region's centre, NOT the starter point. "
        "Version 2 will detect the board edge, estimate the tape centreline, and calculate their intersection."
    )

with tab_lab:
    st.subheader("Image-processing lab")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge_low = st.slider("Canny lower threshold", 0, 255, 60, key="canny_low")
    edge_high = st.slider("Canny upper threshold", 0, 255, 150, key="canny_high")
    edges = cv2.Canny(gray, edge_low, edge_high)
    lab1, lab2 = st.columns(2)
    lab1.image(gray, caption="Grayscale image", clamp=True, use_container_width=True)
    lab2.image(edges, caption="Canny edges", clamp=True, use_container_width=True)

with tab_notes:
    st.subheader("Development roadmap")
    st.markdown(
        """
        **Current — Version 1**
        - Segment pale-yellow tape in HSV colour space.
        - Clean the binary mask with morphological operations.
        - Filter and rank yellow-region contours.
        - Display candidate regions, bounding boxes, and the largest candidate's centre.

        **Next — Version 2**
        - Detect the corrugated-board boundary.
        - Estimate the tape centreline.
        - Calculate the centreline/board-edge intersection (the defined starter point).

        **Later**
        - Validate on a labelled image set under different lighting and camera angles.
        - Add video-frame processing and then a fixed-camera test.
        - Investigate PLC/servo integration only after reliability testing and a formal machine-safety review.
        """
    )
    st.warning(
        "Experimental still-image analysis only. Do not use this prototype to trigger, control, "
        "or alter machinery. It has no PLC, actuator, or live-camera connection."
    )
