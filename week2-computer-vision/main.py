import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VIDEO_PATH = os.path.join(
    PROJECT_DIR,
    "input",
    "football.mp4"
)

YOLO_PATH = os.path.join(
    PROJECT_DIR,
    "yolo11n.pt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "output"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "football_team_colors.mp4"
)


# ============================================================
# ADD src TO PYTHON PATH
# ============================================================

SRC_DIR = os.path.join(
    PROJECT_DIR,
    "src"
)

sys.path.insert(
    0,
    SRC_DIR
)


# ============================================================
# IMPORT YOUR TEAM COLOR FUNCTIONS
# ============================================================

from team_color_assigner import (
    get_player_color,
    get_player_team_id,
    lab_to_bgr,
    draw_boxes
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("PLAYER TEAM COLOR DETECTION")
print("=" * 60)

print("\nSTEP 1: Checking files")

print("Video:")
print(VIDEO_PATH)

print("YOLO:")
print(YOLO_PATH)


# ============================================================
# CHECK VIDEO
# ============================================================

if not os.path.exists(VIDEO_PATH):

    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )

print("\nVideo exists: YES")


# ============================================================
# CHECK YOLO MODEL
# ============================================================

if not os.path.exists(YOLO_PATH):

    raise FileNotFoundError(
        f"YOLO model not found:\n{YOLO_PATH}"
    )

print("YOLO model exists: YES")


# ============================================================
# LOAD VIDEO
# ============================================================

print("\nSTEP 2: Opening video")

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open football.mp4"
    )

print("Video opened: YES")


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print("\nVideo information:")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Frames:", frame_count)

if fps > 0:

    print(
        "Duration:",
        round(frame_count / fps, 2),
        "seconds"
    )


# ============================================================
# LOAD YOLO
# ============================================================

print("\nSTEP 3: Loading YOLO11n")

model = YOLO(
    YOLO_PATH
)

print("YOLO loaded successfully")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# STEP 4: COLLECT PLAYER COLORS
# ============================================================
#
# KMeans needs examples of the colors before it can decide
# which player belongs to which team.
#
# We therefore collect jersey colors from the first part
# of the video.
#
# After collecting enough colors, KMeans learns two clusters:
#
#     Cluster 0 -> Team 1
#     Cluster 1 -> Team 2
#
# ============================================================

print("\nSTEP 4: Collecting jersey colors")

colors = []

COLLECTION_FRAMES = 200

frame_number = 0


while (
    frame_number < COLLECTION_FRAMES
):

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    results = model(
        frame,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # Get detections
    # --------------------------------------------------------

    if result.boxes is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()

    classes = result.boxes.cls.cpu().numpy()

    confidences = result.boxes.conf.cpu().numpy()

    # --------------------------------------------------------
    # Look only for PERSON detections.
    #
    # COCO class 0 = person
    # --------------------------------------------------------

    for box, cls, confidence in zip(
        boxes,
        classes,
        confidences
    ):

        if int(cls) != 0:
            continue

        if confidence < 0.5:
            continue

        xmin, ymin, xmax, ymax = map(
            int,
            box
        )

        # Keep coordinates inside frame
        xmin = max(0, xmin)
        ymin = max(0, ymin)

        xmax = min(width, xmax)
        ymax = min(height, ymax)

        if xmax <= xmin or ymax <= ymin:
            continue

        color = get_player_color(
            frame,
            (xmin, ymin, xmax, ymax)
        )

        if color is not None:

            colors.append(color)

    if frame_number % 20 == 0:

        print(
            f"Collecting frame "
            f"{frame_number}/{COLLECTION_FRAMES} "
            f"- colors collected: {len(colors)}"
        )


# ============================================================
# CHECK COLORS
# ============================================================

print(
    "\nTotal jersey colors collected:",
    len(colors)
)

if len(colors) < 10:

    cap.release()
    out.release()

    raise RuntimeError(
        "Not enough player colors were collected."
    )


# ============================================================
# TRAIN KMEANS
# ============================================================

print("\nSTEP 5: Training KMeans")

colors_array = np.array(
    colors
)

print(
    "Training with",
    len(colors_array),
    "color samples"
)


kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans.fit(
    colors_array
)

print(
    "KMeans trained successfully"
)


# ============================================================
# SHOW LEARNED TEAM COLORS
# ============================================================

print("\nLearned team colors:")

for i, center in enumerate(
    kmeans.cluster_centers_
):

    bgr_color = lab_to_bgr(
        center
    )

    bgr_color = tuple(
        map(
            int,
            bgr_color
        )
    )

    print(
        f"Cluster {i}: "
        f"LAB={center.astype(int)} "
        f"BGR={bgr_color}"
    )


# ============================================================
# RESET VIDEO
# ============================================================

print("\nSTEP 6: Processing complete video")

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

frame_number = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    results = model(
        frame,
        verbose=False
    )

    result = results[0]

    annotated_frame = frame.copy()

    # --------------------------------------------------------
    # Get detections
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        classes = result.boxes.cls.cpu().numpy()

        confidences = result.boxes.conf.cpu().numpy()

        # ----------------------------------------------------
        # Process every person
        # ----------------------------------------------------

        for box, cls, confidence in zip(
            boxes,
            classes,
            confidences
        ):

            # Only persons
            if int(cls) != 0:
                continue

            # Confidence threshold
            if confidence < 0.5:
                continue

            xmin, ymin, xmax, ymax = map(
                int,
                box
            )

            # Keep box inside image
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            xmax = min(width - 1, xmax)
            ymax = min(height - 1, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            player_box = (
                xmin,
                ymin,
                xmax,
                ymax
            )

            # ------------------------------------------------
            # Determine team
            # ------------------------------------------------

            team_id = get_player_team_id(
                frame,
                player_box,
                kmeans
            )

            # ------------------------------------------------
            # Determine display color
            # ------------------------------------------------

            if team_id == 1:

                cluster_color = (
                    kmeans.cluster_centers_[0]
                )

                label = "TEAM 1"

            elif team_id == 2:

                cluster_color = (
                    kmeans.cluster_centers_[1]
                )

                label = "TEAM 2"

            else:

                cluster_color = np.array(
                    [128, 128, 128]
                )

                label = "UNKNOWN"

            # ------------------------------------------------
            # Draw
            # ------------------------------------------------

            annotated_frame = draw_boxes(
                annotated_frame,
                player_box,
                label,
                cluster_color
            )


    # --------------------------------------------------------
    # Frame information
    # --------------------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}/{frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out.write(
        annotated_frame
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "Player Team Colors",
        annotated_frame
    )

    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print(
            "\nQ pressed. Stopping."
        )

        break

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if frame_number % 100 == 0:

        print(
            f"Processed "
            f"{frame_number}/{frame_count}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

out.release()

cv2.destroyAllWindows()


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(
    "Output video:"
)

print(
    OUTPUT_PATH
)

if os.path.exists(OUTPUT_PATH):

    print(
        "Output size:",
        os.path.getsize(OUTPUT_PATH),
        "bytes"
    )

print("=" * 60)