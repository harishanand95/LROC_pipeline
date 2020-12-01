import os
import gdal
from pathlib import Path
import sys


def run(input_filename, out_path, prefix="", count_limit=-1, img_format="PNG", extension=".png"):

    print(input_filename, out_path, count_limit)

    output_filename = 'tile_'

    Path(out_path).mkdir(parents=True, exist_ok=True)

    tile_size_x = 2000
    tile_size_y = 2000

    ds = gdal.Open(input_filename)
    band = ds.GetRasterBand(1)


    xsize = band.XSize
    ysize = band.YSize

    merge_x = 0
    merge_y = 0

    x_inc = tile_size_x
    y_inc = tile_size_y

    print("\n\nSplitting-------------------------------------------------------------------------------\n\n")
    count = 0

    for i in range(0, xsize, tile_size_x):

        x_inc = tile_size_x
        # handle corner case
        if (i + tile_size_x > xsize):
            i = xsize - tile_size_x

        merge_y = 0
        for j in range(0, ysize, tile_size_y):

            y_inc = tile_size_y

            # handle corner case
            if (j + tile_size_y > ysize):
                j = ysize - tile_size_y
            # got scale from gdalinfo -mm filename  and get its min and max val of 1st band
            com_string = "gdal_translate -of {} -srcwin ".format(img_format) + str(i) + ", " + str(j) + ", " + str(x_inc) + ", " + str(
                y_inc) + " " + str(input_filename) + " " + str(out_path) + str(output_filename) + str(
                i) + "_" + str(j) + extension

            os.system(com_string)
            count += 1
            print("Tile : ", i, ",", j)
            if count == count_limit:
                break
            merge_y = merge_y + tile_size_y
        merge_x = merge_x + tile_size_x
        if count == count_limit:
            break

    # com_string = "rm -rf " + out_path + "*.xml"
    # os.system(com_string)
    print("\n\nImage Tiling complete-------------------------------------------------------------------\n\n")



# File and folder name should be NAC_ROI_ALPHNSUSLOA_E129S3581/NAC_ROI_ALPHNSUSLOA_E129S3581.TIF

folder_name = "NAC_ROI_ALPHNSUSLOA_E129S3581"

if not os.path.exists(folder_name + "/split_pngs/"):
    os.makedirs(folder_name + "/split_pngs/")

run('{}/{}.TIF'.format(folder_name, folder_name), 
	'{}/split_pngs/'.format(folder_name), prefix="", count_limit=-1, img_format="PNG", extension=".png")