from datetime import datetime
from utils import *
import cv2
import psycopg
from tqdm import tqdm
import os
from PIL import Image
import rasterio.windows
import rasterio.features
import base64
import json
import requests
import pickle
import pyproj
from tools import *


#NAT_GRID
# 2024-06-16 and 2024-07-15
#JUNE
name1 = './June/'
interval_start = datetime(2023,11,1)
#interval_end = datetime(2024,5,30)

interval_end = datetime(2024,11,1)


#save_folder = './OUTPUTS'

#basemaps_tifs = os.listdir('/cephfs/pimsys/coregistration/basemaps/regions/')

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from pimsys.regions.RegionsDb import RegionsDb
import orbital_vault as ov
#import cosmic_eye_client
from CustomerDatabase import CustomerDatabase
from shapely.geometry import Polygon, MultiPolygon, mapping
import shapely.geometry as geometry

#settings_db = ov.get_sarccdb_credentials()
creds_mapserver = ov.get_image_broker_credentials()

creds = ov.get_sarccdb_credentials()

customer_db_creds = ov.get_customerdb_credentials()
customer_db = CustomerDatabase(customer_db_creds['username'], customer_db_creds['password'])


projects = customer_db.get_projects(active_only=True)
print(len(projects))
for i in range(0, len(projects)):
    print(projects[i]['name'], '-', projects[i]['id'])


#project_name = "National-Fuel-2024" #project_id=54
#project_id = 54
#project_name = "National-Grid-2024"

#project = customer_db.get_project_by_name(project_name)


##### eixa treksei mexri kai slovnaft@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#############################################

with RegionsDb(creds) as database:
    for i in range(18, len(projects)):
        project_name = projects[i]['name']
        project_id = projects[i]['id']    

        print('NOW ', project_name)

        regions = database.get_regions_by_customer(project_name)
        for _, region in enumerate(tqdm(regions)):
            #if region['id'] in [9871815]:

                print(region['id'], project_name)
                xparams = {}

                images = database.get_optical_images_containing_point_in_period([region['bounds'].centroid.x, region['bounds'].centroid.y], [int(interval_start.timestamp()), int(interval_end.timestamp())])  ##can be improved!!
                wms_images = sorted(images, key=lambda x: x["capture_timestamp"])
                wms_images = [x for x in wms_images if x["source"] != "Sentinel-2"]

                if wms_images:
                    target_img = download_from_mapserver(wms_images[0], region['bounds'].bounds, (creds_mapserver['username'], creds_mapserver['password'])) #(1333, 1628, 3)
                    if target_img is not None:

                        clouds1 = predict_clouds_for_model(np.transpose(target_img, (2,0,1))/255., 3, 'https://highrescloudv2-ml.orbitaleye.nl/api/process/rgb') ##1=cloud, 2=haze, 3=shadow

                        if 1 in clouds1:
                            img_patches, mask_patches = cut_patches(target_img, clouds1, 512, 512, 0.2)
                        else:
                            img_patches, mask_patches = [], []

                        if img_patches:
                            for i in range(0, len(img_patches)):
                                img_p = img_patches[i].astype(np.uint8)
                                img_p = Image.fromarray(img_p)
                                img_p.save('./cloud_dataset/images/{}_{}_{}.png'.format(project_id, region['id'], i))

                                mask_p = mask_patches[i]*255
                                #print(np.unique(mask_p))
                                mask_p = Image.fromarray(mask_p)
                                mask_p.save('./cloud_dataset/masks/{}_{}_{}.png'.format(project_id, region['id'], i))
