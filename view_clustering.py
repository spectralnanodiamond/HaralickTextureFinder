import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
#import cv2
import skimage
import tifffile as tf
import os
import pandas as pd
import napari
print(napari.__version__)
from dask_image.imread import imread
import dask.array as da
import io

import platform
if platform.system() == 'Windows':
    fileroot = 'X:/'
    print('Loading Windows')
if platform.system() == 'Linux':
    fileroot = '/home/jovyan/'
    print('Loading Linux')
if False:
    fileroot = '/home/ubuntu/Documents/'
if os.path.exists('/home/bwoodhams/'):
    fileroot = '/home/bwoodhams/'

colors_new=['#0072b2','#d55e00','#009e73', '#cc79a7','#f0e442','#56b4e9']

import sys
sys.path.append(fileroot+'VU_TEX/Ben_utilities/')
sys.path.append(fileroot+'VU_TEX/git/')
import useful_functions as uf
import numba_funcs as nf

from numba import njit
from copy import deepcopy
import matplotlib
from matplotlib.colors import to_rgb, to_rgba
from functools import partial
from importlib import reload
reload(nf)
from tqdm import tqdm

from PIL import Image
from skimage import color
Image.MAX_IMAGE_PIXELS = 1000000000

## Loading colors
#colors = uf.return_color_scale('Omer_chosen_colors_dark_first')
colors_with_inital_grey = deepcopy(colors_new)
colors_with_inital_grey.insert(0, 'gray')
rgb_colors_with_inital_grey = np.asarray(255*np.array([to_rgb(color) for color in colors_with_inital_grey]), int)
print(rgb_colors_with_inital_grey)
dict_colors = {i:color for i, color in enumerate(colors_with_inital_grey)}
print(dict_colors )

rgba_colors_with_inital_grey = np.asarray(255*np.array([to_rgba(color) for color in colors_with_inital_grey]), int)
rgba_colors_with_inital_transparent = rgba_colors_with_inital_grey
rgba_colors_with_inital_transparent[0] = 0
rgba_colors_with_inital_transparent
dict_colors_initial_transp = {i:color for i, color in enumerate(rgba_colors_with_inital_transparent)}
display(dict_colors_initial_transp)
dict_colors_initial_transp_0to1 = {i:color/255 for i, color in enumerate(rgba_colors_with_inital_transparent)}
display(dict_colors_initial_transp_0to1)

dict_gray_out_colors = {0:np.array([0,0,0,0], float), 1:np.array([0,0,0,1], float)}
dict_gray_out_colors

rgb_colors_with_inital_grey = np.asarray(255*np.array([to_rgb(color) for color in colors_with_inital_grey]), int)
print(rgb_colors_with_inital_grey)
dict_colors = {i:color for i, color in enumerate(colors_with_inital_grey)}
print(dict_colors)
rgb_colors_with_inital_grey_0to1 = np.asarray(np.array([to_rgb(color) for color in colors_with_inital_grey]), float)
rgb_colors_with_inital_grey_0to1

## Loading data
#directory_images_metadata = fileroot + 'team283_imaging/' + '0ExternalData/2022-09-01_IVY_GAP/'
#directory_images_metadata = fileroot + '0ExternalData/2022-09-01_IVY_GAP/'
#filename_images_metadata = '2022-09-01_IVY_GAP_metadata.csv'
#df_images_metadata = pd.read_csv(directory_images_metadata + filename_images_metadata, index_col=0)
#df_images_metadata = df_images_metadata.loc[df_images_metadata['original_index'].isin([50, 594, 406, 418, 
#267, 290, 474, 231, 498, 217, 
#309, 544])]
#df_images_metadata

directory_originals = ['../../2023-07-05b_image_conversion_and_cropping/']
filenames_originals = ['V11J11-099__Z4_FO3_01__A1.tif']
fullpaths_originals = [directory_originals[0] + filenames_originals[0]]

df_images_metadata = pd.DataFrame({'original_index': [0], 'output_filename':filenames_originals, 
                                 })
this_shape = skimage.io.imread(fullpaths_originals[0]).shape
df_images_metadata['image_width'] = this_shape[0]
df_images_metadata['image_height'] = this_shape[1]
df_images_metadata


patchsize = 242
this_original_index = 0 #594, 474
#[50, 594, 406, 418, 267, 290, 474, 231, 498, 217, 309, 544] #474
filename_img = df_images_metadata.loc[this_original_index]['output_filename']; print(filename_img)
#filename_gt_original_res = df_images_metadata.loc[this_original_index]['output_filename_annotations']; 
#print(filename_gt_original_res)


original = skimage.io.imread(fullpaths_originals[0])
final_target_size = 1

#directory_colors_metadata = fileroot + '0ExternalData/2022-09-01_IVY_GAP/'
#filename_colors_metadata = '2022-09-02_colors_for_IVY_GAP.csv'
#df_colors = pd.read_csv(directory_colors_metadata + filename_colors_metadata, index_col=0)
#df_colors_out = df_colors.set_index('color_indices').loc[:, ['white_first']].squeeze().to_dict()
#df_colors_out

df_colors = uf.return_color_scale('block_colors_for_labels_against_white_small_points_white_first',
                                  show=False)
df_colors_out = {i:each for i, each in enumerate(df_colors)}
df_colors_out

df_colors_out_no_zero = {key:value for key,value in df_colors_out.items() if key != 0}
df_colors_out_no_zero 


#optional
if False:
    directory_gt_original_res = fileroot + '0ExternalData/2022-09-01_IVY_GAP/2022-09-02_integer_annotations/'
    gt_45 = np.load(directory_gt_original_res + filename_gt_original_res.replace('.jpg', '.npy'))
    print(np.unique(gt_45))
    plt.imshow(gt_45)
    plt.show()
    rgb_gt_45 = uf.label2rgb_with_dict(gt_45, df_colors_out)
    plt.imshow(rgb_gt_45)
    plt.show()

directory_output_masks = 'HDBScan_Output_masks/'
filenames = os.listdir(directory_output_masks)
filenames

from skimage import img_as_ubyte
this_index_filenames = [each for each in filenames if 'Imageindex_'+str(this_original_index) in each]
print(this_index_filenames)
myfilename_ints = [each for each in this_index_filenames if 'ints' in each][0]
print(myfilename_ints)
myfilename_rgb = [each for each in this_index_filenames if 'rgb' in each][0]
print(myfilename_rgb)
intlabels = np.load(directory_output_masks + myfilename_ints)
intlabels -= np.min(intlabels)
plt.imshow(intlabels)
plt.show()
plt.close()
rgblabels = img_as_ubyte(np.load(directory_output_masks + myfilename_rgb))
plt.imshow(rgblabels)
plt.show()
plt.close()


np.unique(intlabels)

int_to_rgb_dict = uf.intlabels_and_rgblabels_to_dict(intlabels, rgblabels, list_to_ignore = [], divide_by_255=True)
int_to_rgb_dict


int_to_rgb_dict_no_zero = uf.intlabels_and_rgblabels_to_dict(intlabels, rgblabels, list_to_ignore = [0], divide_by_255=True)
int_to_rgb_dict_no_zero

intlabels.dtype, np.max(intlabels), np.min(intlabels)

## Napari load
viewer = napari.Viewer()

#original_pyr = uf.get_pyramid_hybrid_loading(original, 2000, 200000)
#original_pyr = [original[::i, ::i] for i in [1, 2, 4, 8, 16, 32]]
#original_pyr = [original[::i, ::i] for i in [1, 2]]
#original_pyr = [np.copy(original[::i, ::i]) for i in [1, 2]]
#original_pyr = [np.copy(original[::i, ::i]) for i in [1, 2, 4, 8, 16, 32]]
original_pyr = uf.get_pyramid_hybrid_loading(original, 1000, 200000)
[each.shape for each in original_pyr]

original_pyr

viewer.add_image(original_pyr, name='Original image')
#viewer.add_image(original, name='Original image')

#optional
if 'gt' in globals():
    gt_pyr = uf.get_pyramid_hybrid_loading(gt, 1000, 200000)
    viewer.add_labels(gt_pyr, name='Groundtruth', visible=False, color=int_to_rgb_dict,)

#optional
if 'gt_45' in globals() and True:
    gt_45_upsize = uf.upsize(gt_45, 45)
    gt_45_pyr = uf.get_pyramid_hybrid_loading(gt_45_upsize, 1000, 200000)
    viewer.add_labels(gt_45_pyr, name='Groundtruth_clusters', visible=False, color=df_colors_out)
    
    output_edges_gt_45 = nf.get_edges_of_cluster_shapes_with_partial_upscale(gt_45, rgb_gt_45, k=6,
                                                                       partial_upscale = 5,
                                                                       final_upscale = 45,
                                                                      image_edges_drawn = True,
                                                                      output_type='int')

    pyr_cluster_edges_gt_45 = uf.get_pyramid_hybrid_loading(output_edges_gt_45, 1000, 200000)

    viewer.add_labels(pyr_cluster_edges_gt_45, color = df_colors_out_no_zero, name='Groundtruth_edges', opacity=1)



@njit
def rgb2labelint_iterator(img, array_of_colors):
    output = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    len_array_colors = len(array_of_colors)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(len_array_colors):
                if np.all(np.equal(array_of_colors[k], img[i,j])):
                    output[i,j] = k
#            output[i,j] = np.argmax(np.all(np.equal(img[i,j], array_of_colors), axis=1))
    return output
    
    
def rgb2labelint(img, array_of_colors = None):
    if array_of_colors is None:
        array_of_colors = np.unique(myimg.reshape(-1, myimg.shape[2]), axis=0)
    output = rgb2labelint_iterator(img, array_of_colors)
    return output

#intlabels = rgb2labelint(myimg, rgb_colors_with_inital_grey)
#plt.figure(figsize=(25,5))
#plt.imshow(intlabels)


partial_upscale = 11
#this the partial upscale needed for creating the edges labels layer
#the higher this number, the smoother the edges are displayed but the slower it takes
final_upscale = int(patchsize / (final_target_size))
    
#This next part grays out the areas that are not part of clusters
if True:
    not_clusters_yet = nf.upsize(intlabels <= 0, patchsize)
    pyr_not_clusters = uf.get_pyramid_hybrid_loading(not_clusters_yet, 1000, 2000000)
    viewer.add_labels(pyr_not_clusters, color=dict_gray_out_colors, 
                      name='Part of any cluster', visible=False)


#This next part creates a labels layer that shows the edges of the clusters
if True:
#    rgbalabels = nf.convert_rgb_to_rgba(rgblabels, transparent=np.array([255, 255, 255]), transparency_val = 255)
    output_edges = nf.get_edges_of_cluster_shapes_with_partial_upscale(intlabels, rgblabels, k=2,
                                                                       partial_upscale = partial_upscale,
                                                                       final_upscale = final_upscale,
                                                                      image_edges_drawn = True,
                                                                      output_type='int')
#    intlabels_partial_upsize = nf.upsize(intlabels, partial_upscale)
#    rgblabels_partial_upsize = nf.upsize2(rgblabels, partial_upscale)

#    output_edges = nf.get_edges_of_cluster_shapes(intlabels_partial_upsize, rgblabels_partial_upsize, k=4)
#    output_edges_labels_for_int_layer = rgb2labelint(output_edges, rgb_colors_with_inital_grey)    
#    output_edges_labels_for_int_layer_fullsize = nf.upsize(output_edges_labels_for_int_layer, further_upscale)

#    pyr_cluster_edges = uf.get_pyramid_hybrid_loading(output_edges_labels_for_int_layer_fullsize, 2000, 2000000)
#    pyr_cluster_edges = [output_edges[::i, ::i] for i in [1, 2, 4, 8, 16, 32]]
    pyr_cluster_edges = uf.get_pyramid_hybrid_loading(output_edges, 1000, 200000)
#    viewer.add_labels(pyr_cluster_edges, color=dict_colors_initial_transp_0to1, 
#                      name='Cluster edges', visible=False, opacity=1)
    viewer.add_labels(pyr_cluster_edges, color = int_to_rgb_dict_no_zero, name='Our edges', opacity=1)

#This next part shows the clusters as shaded patches
if True:
    intlabels_upsized = nf.upsize(intlabels, final_upscale)
    intlabels_pyr = uf.get_pyramid_hybrid_loading(intlabels_upsized, 1000, 200000)
    viewer.add_labels(intlabels_pyr, color=int_to_rgb_dict, 
                      name='Our clusters', opacity=0.2, visible=False)
#    cluster_patches_int = rgb2labelint(rgblabels_partial_upsize, rgb_colors_with_inital_grey)
#    cluster_patches_int_fullsize = nf.upsize(cluster_patches_int, further_upscale)
#    img_clusters = uf.get_pyramid_hybrid_loading(cluster_patches_int_fullsize, 2000, 2000000)
#    viewer.add_labels(img_clusters, color=dict_colors_initial_transp_0to1, 
#                      name='Shaded clusters', opacity=0.2)

napari.run()