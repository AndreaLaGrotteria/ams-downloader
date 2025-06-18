import cv2
import numpy as np
import os 
from tqdm import tqdm

# creates 6 90 degree cubes from the 360 panorama 
def equirec_to_cubemap(equirec, out_size, car_heading):
    height, width, _ = equirec.shape
    print(f"Input shape: {equirec.shape}")

    # grid the images 
    u, v = np.meshgrid(np.linspace(-1, 1, out_size), np.linspace(-1, 1, out_size))
    ones = np.ones((out_size, out_size), dtype=np.float32)

    # define face directions 
    list_xyz = [
        (ones, v, -u),   # 1
        (u, v, ones),    # 2
        (-ones, v, u),   # 3
        (-u, v, -ones),  # 4
        (u, ones, v),    # 5
        (u, -ones, -v)   # 6
    ]

    faces = []

    # convert car heading from degrees to radians
    car_heading_rad = np.radians(car_heading)
    
    # images need to be dewarped to get rid of the distortion
    for i, (x, y, z) in enumerate(list_xyz):

        #normalize
        r = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x/r, y/r, z/r

        # rotate horizontal faces based on car heading leave vertical alone
        if i < 4:  
            x_rot = x * np.cos(car_heading_rad) + z * np.sin(car_heading_rad)
            z_rot = -x * np.sin(car_heading_rad) + z * np.cos(car_heading_rad)
        else:
            x_rot, z_rot = x, z

        # convert to spherical coordinates
        theta = np.arctan2(x_rot, z_rot)
        phi = np.arcsin(y)

        # map to equirectangular coordinates
        phi_map = ((phi / np.pi + 0.5) * height).astype(np.float32)
        theta_map = ((theta / (2 * np.pi) + 0.5) * width).astype(np.float32)

        # clip to image boundaries
        theta_map = np.clip(theta_map, 0, width - 1)
        phi_map = np.clip(phi_map, 0, height - 1)

        # remap the equirectangular image to the cubemap face
        face = cv2.remap(equirec, theta_map, phi_map, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        faces.append(face)

    return faces