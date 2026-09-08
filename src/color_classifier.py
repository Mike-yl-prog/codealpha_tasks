import cv2
import numpy as np


def get_player_color(frame, bbox):
    x1, y1, x2, y2 = bbox

    # crop the player from the frame
    player_crop = frame[y1:y2, x1:x2]

    if player_crop.size == 0:
        return None

    height, width = player_crop_shape[:2]

    # focus on the upper-middle part of the player
    # where the jersry is likely to be.
    jersey_crop = player_crop [
        int(height * 0.20):int(height * 0.60),
        int(width * 0.20):int(width * 0.80)
    ]

    if jersey_crop.size == 0:
        return None

    #convert BGR -> RGB
    jersey_rgb = cv2.cvColor(jersey_crop, cv2.COLOR_BGR2RGB)

    # calculate the average color
    average_color = np.mean(
        jersey_rgb.reshape(-1, 3),
        axis=0
    )

    return average_color
    

