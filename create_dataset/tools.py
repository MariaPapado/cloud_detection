import rasterio
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi



def save_tif_coregistered_with_params(filename, image, xparams, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    with rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform) as dst:
   # Write bands
        if channels>1:
         for ch in range(0, image.shape[2]):
           dst.write(image[:,:,ch], ch+1)
        else:
           dst.write(image, 1)

        dst.update_tags(**xparams)
#        dst.update_tags(img_id_after='{}'.format(xparams['id_before']))
        
    dst.close()

    return True



def save_tif_coregistered(filename, image, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
    if channels>1:
     for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)
    else:
       new_dataset.write(image, 1)
    new_dataset.close()

    return True

 
def visualize(dam):
    dam = np.array(dam, dtype=np.uint8)
    dam = Image.fromarray(dam)
    dam.putpalette([0, 0, 0,
                    0, 255, 0,
                    255, 0, 0,
                    128, 128, 128])
    dam = dam.convert('RGB')
    dam = np.asarray(dam)

    return dam
 
