import cv2
import numpy as np


# ============================================================
# COLOR REPRESENTATION
# ============================================================
# OpenCV normally represents images using BGR:
#
#     Blue  Green  Red
#
# For jersey-color classification, we convert the image to LAB.
#
# LAB separates:
#
#     L -> lightness
#     A -> green <-> red
#     B -> blue  <-> yellow
#
# This makes LAB useful when we want to compare colors while
# reducing the influence of brightness.
# ============================================================

def lab_to_bgr(lab_color):
    """
    Convert one LAB color back into BGR.

    OpenCV drawing functions such as cv2.rectangle() expect
    colors in BGR format, so we convert the learned jersey
    color back before drawing it on the video frame.
    """

    # OpenCV expects an 8-bit LAB image.
    # We create a tiny image containing only one pixel.
    lab_img = np.uint8([[lab_color]])

    # Convert that single LAB pixel into BGR.
    bgr = cv2.cvtColor(
        lab_img,
        cv2.COLOR_LAB2BGR
    )

    # Return the single BGR pixel.
    return bgr[0][0]


# ============================================================
# VISUALIZATION
# ============================================================
# This function is responsible only for displaying the result.
#
# It does NOT decide which team a player belongs to.
# It receives the result of the classification and draws it.
# ============================================================

def draw_boxes(image, box, label, color):

    # The classifier works in LAB, but OpenCV drawing
    # functions expect BGR.
    color = lab_to_bgr(color)

    # Convert NumPy values into normal Python integers because
    # OpenCV drawing functions expect integer color values.
    color = tuple(map(int, color))

    # Extract the player's bounding-box coordinates.
    xmin, ymin, xmax, ymax = box

    # Draw the player's bounding box.
    cv2.rectangle(
        image,
        (xmin, ymin),
        (xmax, ymax),
        color,
        2
    )

    # --------------------------------------------------------
    # Prepare the text label.
    # --------------------------------------------------------

    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        3
    )

    tw, th = text_size

    # Put the label above the player's bounding box.
    # max(..., 0) prevents the text from going outside
    # the top of the image.
    ymin_text = max(ymin - th - 5, 0)

    # Draw a colored background behind the label so that
    # the text remains readable.
    cv2.rectangle(
        image,
        (xmin, ymin_text - 5),
        (xmin + tw + 10, ymin),
        color,
        -1
    )

    # Draw the actual team/player label.
    cv2.putText(
        image,
        label,
        (xmin + 5, ymin - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )

    return image


# ============================================================
# PLAYER JERSEY COLOR EXTRACTION
# ============================================================
# A player bounding box contains much more than the jersey:
#
#     head
#     arms
#     legs
#     grass
#     background
#
# We therefore don't use the entire bounding box to determine
# the team color.
#
# Instead, we take an approximate chest region where the jersey
# is most likely to be visible.
# ============================================================

def get_player_color(frame, bbox):

    # Extract bounding-box coordinates.
    xmin, ymin, xmax, ymax = bbox

    # Crop the player from the complete video frame.
    #
    # NumPy image slicing uses:
    #
    #     image[y1:y2, x1:x2]
    #
    # Notice that Y comes before X.
    player_img = frame[
        ymin:ymax,
        xmin:xmax
    ]

    # A detection can occasionally produce an invalid or empty
    # crop. We handle that case instead of allowing OpenCV to
    # crash later.
    if player_img.size == 0:
        return None

    # Get dimensions of the player's cropped image.
    h, w = player_img.shape[:2]

    # --------------------------------------------------------
    # Extract approximate chest region.
    #
    # We intentionally avoid the top of the body because it may
    # contain the player's face/head and avoid the bottom because
    # it may contain shorts, legs, or grass.
    # --------------------------------------------------------

    chest = player_img[
    int(h * 0.25):int(h * 0.55),
    int(w * 0.25):int(w * 0.75)
]



    if chest.size == 0:
        return None

    # --------------------------------------------------------
    # Convert the chest from BGR to LAB.
    #
    # The original frame comes from OpenCV in BGR format.
    # LAB gives us a color representation that is more useful
    # for comparing jersey colors.
    # --------------------------------------------------------

    chest_lab = cv2.cvtColor(
        chest,
        cv2.COLOR_BGR2LAB
    )

    # --------------------------------------------------------
    # Convert the 2D image into a list of pixels.
    #
    # Before:
    #
    #     height × width × 3
    #
    # After reshape:
    #
    #     number_of_pixels × 3
    #
    # Each row now represents:
    #
    #     [L, A, B]
    # --------------------------------------------------------

    pixels = chest_lab.reshape(-1, 3)

    # Instead of choosing one random pixel, calculate the
    # median LAB value across the chest region.
    #
    # The median is useful because individual pixels may contain
    # noise, skin, shadows, logos, grass, or other colors.
    color = np.median(
        pixels,
        axis=0
    )

    # Return the representative jersey color.
    return color


# ============================================================
# TEAM CLASSIFICATION
# ============================================================
# At this point we have:
#
#     player image
#          ↓
#     chest region
#          ↓
#     LAB pixels
#          ↓
#     representative LAB color
#
# The KMeans model then decides which learned color cluster
# the player belongs to.
#
# For example:
#
#     Cluster 0 -> Team A
#     Cluster 1 -> Team B
#
# The actual cluster numbers do NOT inherently mean "Team A"
# or "Team B". They are simply labels produced by KMeans.
# ============================================================

def get_player_team_id(frame, box, model):

    # Extract the player's representative jersey color.
    color = get_player_color(
        frame,
        box
    )

    # If we couldn't extract a valid color, return 0 to represent
    # an unknown/unclassified player.
    if color is None:
        return 0

    # --------------------------------------------------------
    # KMeans expects a 2D array:
    #
    #     samples × features
    #
    # Our color is currently:
    #
    #     [L, A, B]
    #
    # Therefore reshape it into:
    #
    #     [[L, A, B]]
    #
    # which represents one sample with three features.
    # --------------------------------------------------------

    prediction = model.predict(
        color.reshape(1, -1)
    )

    # Extract the cluster number from the prediction.
    team_id = int(prediction[0])

    # We add 1 because 0 is reserved for "unknown".
    #
    # Therefore:
    #
    #     0 -> unknown
    #     1 -> team cluster 0
    #     2 -> team cluster 1
    #
    return team_id + 1
