import numpy as np
import cv2
import requests
from geopy.distance import geodesic
from shapely import geometry, wkb
from datetime import datetime
import torch
import clip
from PIL import Image as imp
from typing import Dict, Any
import rasterio
import rasterio.features
import base64
import json
import pyproj
from shapely import ops
from shapely.geometry import Polygon, box
from pimsys.regions.RegionsDb import RegionsDb


def tile_image(mask, tile_size, tile_buffer):
    img_x = []
    for tile in np.arange(np.ceil(image.shape[1] / tile_size)):
        tile_start = tile * (tile_size - tile_buffer)
        tile_end = tile_start + tile_size
        if tile_end > image.shape[1]:
            if image.shape[1] - tile_size < 0:
                img_x.append(0)
            else:
                img_x.append(image.shape[1] - tile_size)
        else:
            img_x.append(tile_start)

    img_y = []
    for tile in np.arange(np.ceil(image.shape[2] / tile_size)):
        tile_start = tile * (tile_size - tile_buffer)
        tile_end = tile_start + tile_size
        if tile_end > image.shape[2]:
            if image.shape[2] - tile_size < 0:
                img_y.append(0)
            else:
                img_y.append(image.shape[2] - tile_size)
        else:
            img_y.append(tile_start)

    return img_x, img_y



def cut_patches(img, mask, p_size, step, perc):
    height, width = mask.shape[0], mask.shape[1]

    img_patches = []
    mask_patches = []
    img, mask = np.array(img), np.array(mask)
    for x in range(0, height, step):
        #print(image_before.tile_size)
        if x + p_size > height:
            x = height - p_size
        for y in range(0, width, step):
            if y + p_size > width:
                y = width - p_size
            mask_patch = mask[x:x+p_size, y:y+p_size]
            idx1 = np.where(mask_patch==1)
            ratio = len(idx1[0])/(p_size*p_size)
#            print(ratio)
            if ratio>perc:
                print(ratio)
                img_patches.append(img[x:x+p_size, y:y+p_size,:])
                idx_else = np.where(mask_patch!=1)
                mask_patch[idx_else]=0
                mask_patches.append(mask_patch)
    return img_patches, mask_patches


def predict_clouds_for_model(image, bands, test_url_cloud):
    image = np.moveaxis(image[:bands], 0, -1)
    image_d = base64.b64encode(np.ascontiguousarray(image.astype(np.float32)))

    response = requests.post(test_url_cloud,json={"image": image_d.decode(), "shape": json.dumps(list(image.shape))})

    if response.ok:
        response_result = json.loads(response.text)
        response_result_data = base64.b64decode(response_result["result"])
        result = np.frombuffer(response_result_data, dtype=np.uint8)
        cloud_mask = result.reshape(image.shape[:2])

    else:
        print("error", response)
        raise ValueError("error wrong response from cloud server")
    return cloud_mask




def download_from_mapserver(image, region_bounds, auth=None):
    pixel_resolution_x = 0.5#layer["pixel_resolution_x"]
    pixel_resolution_y = 0.5#layer["pixel_resolution_y"]

    region_width = geodesic(
    (region_bounds[1], region_bounds[0]), (region_bounds[1], region_bounds[2])
    ).meters
    region_height = geodesic(
    (region_bounds[1], region_bounds[0]), (region_bounds[3], region_bounds[0])
    ).meters

    width = int(round(region_width / pixel_resolution_x))
    height = int(round(region_height / pixel_resolution_y))

    url = "https://maps.orbitaleye.nl/mapserver/?map=/maps/_"
    image_url = url + f'{image["wms_layer_name"]}.map&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX={region_bounds[0]},{region_bounds[1]},{region_bounds[2]},{region_bounds[3]}&WIDTH={width}&HEIGHT={height}&FORMAT=image/png&LAYERS={image["wms_layer_name"]}'
    resp = requests.get(image_url, auth=auth)
    if resp.ok:
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is not None:
            image = image[:, :, [2, 1, 0]]
    else: 
        print(image_url)
        return None
    
    return image







