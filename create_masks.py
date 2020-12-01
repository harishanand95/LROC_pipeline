from PIL import Image
import numpy as np
import torch
from model import get_model_instance_segmentation
import os
import glob
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr


import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg



def generate_mask(img_base, img_name):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    num_classes = 2
    image_mean = [0.34616187074865956, 0.34616187074865956, 0.34616187074865956]
    image_std = [0.10754412766560431, 0.10754412766560431, 0.10754412766560431]

    mask_rcnn = get_model_instance_segmentation(num_classes, image_mean, image_std, stats=True)
    mask_rcnn.to(device)
    mask_rcnn.eval()
    model_path = "models/epoch_0099.param"
    mask_rcnn.load_state_dict(torch.load(model_path))

    # tif files and png files location
    img_path = img_base + "/split_pngs/"
    tif_path = img_base + "/split_tifs/"


    # tmp folder has the vector files created for each craters
    # create if it doesn't exist
    tmp_folder = img_base + "/split_tifs/tmp/"
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    pred_folder = img_base + "/split_tifs/shape_files/"
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    print(img_name)
    # file name without extension
    file_name = img_name.split(".")[0]
    
    individual_folder = tmp_folder + file_name + "/"
    if not os.path.exists(individual_folder):
        os.makedirs(individual_folder)

    image = Image.open(img_path + img_name).convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image / 255.0).float()
    image = image.permute((2, 0, 1))
    pred = mask_rcnn(image.unsqueeze(0).to(device))[0]

    #from visualize import display_instances
    boxes_ = pred["boxes"].cpu().detach().numpy().astype(int)
    boxes = np.empty_like(boxes_)
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = boxes_[:, 1], boxes_[:, 0], boxes_[:, 3], boxes_[:, 2]
    labels = pred["labels"].cpu().detach().numpy()
    scores = pred["scores"].cpu().detach().numpy()
    masks = pred["masks"]

    indices = scores > 0.6

    # Show Scores
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]

    masks = masks[indices].squeeze(1)
    masks = (masks.permute((1, 2, 0)).cpu().detach().numpy() > 0.5).astype(np.uint8)

    # Show image predictions and save as npy files
    image = image.permute((1, 2, 0)).cpu().detach().numpy() * 255
    #display_instances(image, boxes, masks, labels, class_names=["background", "crater"], scores=scores)


    print(masks.shape)

    # Save masks as tif files
    crater_prediction_count = masks.shape[2]


    for i in range(crater_prediction_count):
        # create single crater tif file
        tmp_tif = individual_folder + file_name + "_" + str(i) + ".tif"

        os.system("cp " + tif_path + file_name + ".tif " + tmp_tif)
        dataset = gdal.Open(tmp_tif, gdal.GA_Update)
        band = dataset.GetRasterBand(1)
        new_mask = masks[:, :, i]
        new_mask = new_mask
        band.SetNoDataValue(0)
        band.WriteArray(new_mask)
        dataset.FlushCache()
        del dataset

        # convert the tif file to a vector data gdal polygonize
        tmp_shp = individual_folder + file_name + "_" + str(i) + ".shp"
        os.system("""gdal_polygonize.py {} {} -b 1 -f "ESRI Shapefile" 1 DN """.format(tmp_tif, tmp_shp))
    
    if crater_prediction_count > 0:
        os.system("ogrmerge.py -single -overwrite_ds -o {} {}*.shp".format(pred_folder + file_name + ".shp", individual_folder))
        os.system("rm -rf " + individual_folder + "*")


folder_name = "NAC_ROI_ALPHNSUSLOA_E129S3581"


# Test prediction on a single png file
TEST_ON_ONE_FILE = False
if TEST_ON_ONE_FILE:
    generate_mask(folder_name, "tile_0_0.png", mask_rcnn, device)
    exit()



# LROC images should be split into PNG files using split_tifs.py file
image_files = os.listdir(folder_name + '/split_pngs/')

## Multiprocessing Pool
import multiprocessing
p = multiprocessing.Pool(processes=30)

for image_file in image_files:
    if ".png" in image_file[-4:]:
        p.apply_async(generate_mask, [folder_name, image_file]) 
        # generate_mask(folder_name, image_file)

p.close()
p.join() # Wait for all child processes to close
exit()
# Load the png files and predict its mask
# Save the masks as tif files in predicted_masks/ folder
for image_file in image_files:
    if ".png" in image_file[-4:]:
        generate_mask(folder_name, image_file, mask_rcnn, device)

exit()

# Convert the mask tifs to a single tif "out.tif" file using gdal merge
os.system("gdal_merge.py -o out.tif predicted_masks/*.tif")


# Convert the out.tif file to a vector format (Polygonize a raster file using GDAL)
os.system("""gdal_polygonize.py predicted_masks/out.tif shape_files/out.geojson -b 1 -f "GeoJSON" out DN""")

