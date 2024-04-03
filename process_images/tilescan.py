import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
from PIL import Image
import csv
import cv2
from skimage.transform import resize
import numpy as np
import pandas as pd
import itertools
import argparse
import sys
from mmdet.apis import init_detector, inference_detector
import mmcv
import threading
import concurrent.futures
import traceback

parser = argparse.ArgumentParser(description = 'OBB-CNN Process Images for Object Extraction')
parser.add_argument('-i', '--input', type=str, help='Directory for images to run inference on (default = input).')
parser.add_argument('-m', '--model', type=str, help='Directory for model/weights file to run inference with (default = models/epoch_1000.pth).')
parser.add_argument('-c', '--config', type=str, help='Directory for config file to run inference with (default = config/mrcnn_r152.py).')
parser.add_argument('-o', '--output', type=str, help='Directory for image cutouts to be stored (default = output).')
parser.add_argument('-r', '--results', type=str, help='Directory for quantifications to be stored (default = results).')
parser.add_argument('-s', '--size', type=int, help='Size of tiles to process (default = 1024).')
parser.add_argument('-p', '--per', type=float, help='Percent overlap of tiles for processing (default = 0.25).')
args = parser.parse_args()
    
Image.MAX_IMAGE_PIXELS = None

m = f'{args.model}'

if args.model == None:
    m = 'models/epoch_1000.pth'
    
c = f'{args.config}'

if args.config == None:
    c = 'config/mrcnn_r152.py'

s = f'{args.input}'

if args.input == None:
    s = 'input/'
    
sources = [j for j in os.listdir(s) if j.rpartition('.')[2] in ('png', 'tif', 'jpg')]

r = f'{args.results}'

if args.results == None:
    r = 'results/'
    
if not os.path.exists(r):
    os.makedirs(r)
    results = []
else:
    results = [os.path.splitext(os.path.basename(j))[0] for j in os.listdir(r) if j.rpartition('.')[2] in ('csv')]
    
o = f'{args.output}'

if args.output == None:
    o = 'output/' 

if not os.path.exists(o):
    os.makedirs(o)
    
global size

size = f'{args.size}'

if args.model == None:
    size = 1024
    
size = int(size)
    
global per
    
per = f'{args.per}'

if args.per == None:
    per = 0.25

# build the model from a config file and a checkpoint file
model = init_detector(c, m, device='cuda:0')
# threshold for mask prediciton probability
score_thr = 0.1
# threshold for overlap percentage for merging masks
threshold = 0.6
    
def analyze(source):
    
    try:
        
        name = os.path.splitext(os.path.basename(source))[0]
        sname = source.split(" ")

        oimage = cv2.imread(os.path.join(s, source))
        oa, ob = oimage.shape[:2]
        #rounds border size up to create an even number of tiles when dividing image
        ba = (size) - (oa % (size))
        bb = (size) - (ob % (size))

        ma = oa + ba
        mb = ob + bb
        wbg =  np.ones((ma, mb, 3), dtype=np.uint8)*255

        wbg[0:oa, 0:ob] = oimage
        image = wbg
        
        dfs = process_image(image, image, ma, mb, sname)

        df = pd.DataFrame()
        df = pd.concat(dfs)
        rfp = os.path.join(r, f'{name}.csv')

        if os.path.exists(rfp):
            df.to_csv(rfp, mode='a', header=False, index=False)
        else:
            df.to_csv(rfp, index=False)
    except Exception as e:
        traceback.print_exc()

            
def process_image(image, wbg, ma, mb, sname):
    
    allmasks = []

    height, width = image.shape[:2]

    stitch = np.zeros((height, width), dtype=np.uint8)
    
    overlap = int(per*size)

    global t_width
    t_height = (height - size) // overlap + 1
    t_width = (width - size) // overlap + 1

    for row in range(t_height):
        
        for col in range(t_width):
            
            start_x = col * overlap
            start_y = row * overlap
            end_x = start_x + size
            end_y = start_y + size

            if end_x > width:
                
                start_x = width - size
                end_x = width
                
            if end_y > height:
                
                start_y = height - size
                end_y = height

            tile = image[start_y:end_y, start_x:end_x]
            
            masks, tile_mask = mask_gen(tile, start_y, start_x)
            '''
            # Stitch processed tile onto background (for color normalization)
            stitch[start_y:end_y, start_x:end_x] = blend_images(stitch[start_y:end_y, start_x:end_x], tile_mask)
            '''
            allmasks.append(masks)
    
    dfs = merge_masks(allmasks, height, width, sname, wbg, ma, mb) #Removed stitch (required for normalizing color)
    
    return dfs
    
    
def mask_gen(tile, start_y, start_x):
    
    masks = []
    #adds blur to image to reduce noise
    tile = cv2.blur(tile, (5, 5))

    result = inference_detector(model, tile)

    bbox_result, segm_result = result
    bboxes = np.vstack(bbox_result)
    
    # draw segmentation masks
    if segm_result is not None:
        
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        
        for ids in inds:
            
            ids = int(ids)
            mk = segms[ids]
            mk = mk.astype(np.uint8)
            
            wt = np.sum(mk)
            copy = np.array(mk, copy=True)
            if wt > 500:
                masks.append((copy, start_y, start_x))
        
        #merge duplicate masks in the same tile and make binary mask of whole image
        tile_mask = np.zeros((size, size), dtype=np.uint8)
        
        if len(masks) > 0:
            
            for i in reversed(range(len(masks))):
                binary_mask = masks[i][0].astype(np.uint8)
                tile_mask = np.logical_or(tile_mask, binary_mask)

                for j in reversed(range(len(masks))):

                    if i != j and i < len(masks):

                        intersection = np.logical_and(masks[i][0], masks[j][0])

                        overlap = np.sum(intersection)

                        overlap1 = overlap / np.sum(masks[i][0])
                        overlap2 = overlap / np.sum(masks[j][0])

                        if overlap1 > 0.8 or overlap2 > 0.8 and overlap > 0:
                            newmask = np.logical_or(masks[i][0], masks[j][0])
                            masks[i] = (newmask, masks[i][1], masks[i][2])
                            
                            del masks[j]
        
    return masks, tile_mask


def merge_masks(allmasks, height, width, sname, wbg, ma, mb): #Removed stitch (required for normalizing color)
    
    dfs = []
    
    p = 1

    for t in range(len(allmasks)):
        
        if len(allmasks[t]) > 0:
        
            for i, tt in reversed(list(enumerate(allmasks[t]))):
                
                ti = np.zeros((height, width), dtype=np.uint8)
                ti[tt[1]:(tt[1]+size), tt[2]:(tt[2]+size)] = tt[0]
                
                delete = [(t, i)]
                
                directions = neighbors(tt, t)
                
                allmasks, find, finds = seek(allmasks, ti, directions, delete, height, width)
                
                mpix = np.array(find)
                maskpix = np.sum(mpix == 1)

                if maskpix > 25000 and finds >= 5: #filters out all objects found that do not exceed a specific size and number of merged tiles
                
                    df_obj = process(find, p, sname, wbg, ma, mb) #Removed stitch (required for normalizing color)
                    
                    dfs.append(df_obj)

                    p += 1
    
    return dfs


def seek(allmasks, ti, directions, delete, height, width):
    
    done = False
    found = True
    finds = 0
    find = ti
    
    while not done:

        if not found:
            done = True
        
        found = False

        for e in directions:

            pr = e[0]
            p = e[1]
            direct = e[2]
            direc = direct[0]
            mult = direct[1]
            base = direct[2]
            
            d = (p + direc)

            if d in range(len(allmasks)) and len(allmasks[d]) > 0:

                for c, dd in reversed(list(enumerate(allmasks[d]))):
                    
                    if (d, c) not in delete:
                        
                        dc = np.zeros((height, width), dtype=np.uint8)
                        dc[dd[1]:(dd[1]+size), dd[2]:(dd[2]+size)] = dd[0]

                        if base == 1:
                            pcrop = pr[0][0:size,int(mult*size):size]
                            dcrop = dd[0][0:size,0:int((1-mult)*size)]
                        elif base == -1:
                            pcrop = pr[0][0:size,0:int((1-mult)*size)]
                            dcrop = dd[0][0:size,int(mult*size):size]
                        elif base == t_width:
                            pcrop = pr[0][int(mult*size):size,0:size]
                            dcrop = dd[0][0:int((1-mult)*size),0:size]
                        elif base == -t_width:
                            pcrop = pr[0][0:int((1-mult)*size),0:size]
                            dcrop = dd[0][int(mult*size):size,0:size]
                            
                        intersection = np.logical_and(pcrop, dcrop)

                        overlap = np.sum(intersection)

                        overlap1 = overlap / np.sum(pcrop)
                        overlap2 = overlap / np.sum(dcrop)

                        if overlap1 > threshold and overlap2 > threshold and overlap > 0:

                            found = True

                            find = np.logical_or(find, dc)

                            directions.extend(neighbors(dd, d))

                            delete.append((d, c))
                            
                            finds += 1
        
    delete = sorted(list(set(delete)), key = lambda x:x[1], reverse = True)
    for t in delete:
        try:
            del allmasks[t[0]][t[1]]
        except:
            pass
        
    return allmasks, find, finds

'''
def blend_images(stitch, tile):
    
    # Create a mask of the white pixels
    mask = (tile >= 1)

    # blend image while retaining whitespace
    blended_image = np.where(mask, tile, stitch)

    return blended_image
'''

def get_multiples(per):
    multiples = [[per, 1]]
    counts = [1]
    count = 1
    current = per

    while current < 0.7:
        count += 1
        current += per
        if current <= 0.7:
            multiples.append([round(current, 2), count])
            
    return multiples

def neighbors(dd, d):
    
    nb = []
    direcs = []
    
    multiples = get_multiples(per)
    
    for mult in multiples:    
        direcs.append([1*mult[1], mult[0], 1])
        direcs.append([-1*mult[1], mult[0], -1])
        direcs.append([t_width*mult[1], mult[0], t_width])
        direcs.append([-t_width*mult[1], mult[0], -t_width])
    
    for direc in direcs:
        nb.append([dd, d, direc])
        
    #check = 100 #Checks for whitespace in _ pixel border from edge (can potentially improve efficiency)
    
    #crop = dd[0:size, size-check:size]
    #if np.sum(crop) > 100:

    #crop = dd[0:size, 0:check]
    #if np.sum(crop) > 100:

    #crop = dd[size-check:size, 0:size]
    #if np.sum(crop) > 100:

    #crop = dd[0:check, 0:size]
    #if np.sum(crop) > 100:
    
    return nb


def process(find, p, sname, wbg, ma, mb): #Removed stitch (required for normalizing color)
    
    slide = sname[5].split('.')
    mask = find.astype(np.uint8)
    #mask = cv2.resize(mask, (mb, ma), interpolation = cv2.INTER_AREA)
    #stitch = cv2.resize(stitch, (mb , ma), interpolation = cv2.INTER_AREA)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:

        ratio = area/perimeter

    a, b, w, h = cv2.boundingRect(contour)
    cmask = mask[b:b+h, a:a+w]
    cut = wbg[b:b+h, a:a+w]
    #cstitch = stitch[b:b+h, a:a+w]
    cutout = cut * cmask[:, :, np.newaxis]
    
    #saves image of identified object and its mask
    cv2.imwrite(os.path.join(o, f'cutout{sname[0]}_{slide[0]}_{p}.png'), cutout)
    cv2.imwrite(os.path.join(o, f'mask{sname[0]}_{slide[0]}_{p}.png'), cmask*255)
    
    '''
    #Normalize by backgroud color (optional):
    ref_r = np.mean(cut[:, :, 0][cstitch == 0])
    ref_g = np.mean(cut[:, :, 1][cstitch == 0])
    ref_b = np.mean(cut[:, :, 2][cstitch == 0])
    
    norm_cutout_r = cut[:, :, 0].astype(np.float32) / ref_r.astype(np.float32)
    norm_cutout_r = np.clip(norm_cutout_r, 0, 1)
    norm_cutout_g = cut[:, :, 1].astype(np.float32) / ref_g.astype(np.float32)
    norm_cutout_g = np.clip(norm_cutout_g, 0, 1)
    norm_cutout_b = cut[:, :, 2].astype(np.float32) / ref_b.astype(np.float32)
    norm_cutout_b = np.clip(norm_cutout_b, 0, 1)
    norm_cutout = cv2.merge((norm_cutout_b, norm_cutout_g, norm_cutout_r))
    norm_cutout = (norm_cutout * 255 * cmask[:, :, np.newaxis]).astype(np.uint8)
    '''
    
    gray = cv2.cvtColor(cutout, cv2.COLOR_BGR2GRAY)

    # thresholding to separate "dark" and "light" regions (can be also used to separate out specific colors i.e. brown for agouti hair)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    dark = np.count_nonzero(thresh[cmask != 0] == 0)
    light = np.count_nonzero(thresh[cmask != 0] == 255)
    dlr = dark / light

    avg_red = np.mean(cut[:, :, 0][cmask != 0])
    avg_green = np.mean(cut[:, :, 1][cmask != 0])
    avg_blue = np.mean(cut[:, :, 2][cmask != 0])
    
    std_red = np.std(cut[:, :, 0][cmask != 0])
    std_green = np.std(cut[:, :, 1][cmask != 0])
    std_blue = np.std(cut[:, :, 2][cmask != 0])
    
    #defines all measured parameters
    df_obj = pd.DataFrame({'ID': sname[0],
                    'Age': sname[1],
                    'Condition': sname[2],
                    'Sex': sname[3],
                    'Color': sname[4],
                    'Slide': slide[0],
                    'Object' : p,
                    'Area': area,
                    'Perimeter': perimeter,
                    'A/P': ratio,
                    'Red': avg_red,
                    'Green': avg_green,
                    'Blue': avg_blue,
                    'Red SD': std_red,
                    'Green SD': std_green,
                    'Blue SD': std_blue,
                    'D/L': dlr}, index = [0])
    
    return df_obj


def thread(sources):
    #multithreads to process multiple images simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for start in range(len(sources)):
            if os.path.splitext(os.path.basename(sources[start]))[0] not in results:
                futures.append(executor.submit(analyze, sources[start]))
            else:
                pass
        concurrent.futures.wait(futures)

thread(sources)
