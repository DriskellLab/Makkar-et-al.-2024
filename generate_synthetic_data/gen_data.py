import sys
import os
import random
import numpy as np
import cv2
from PIL import Image
import json
import queue
import threading
import concurrent.futures
import argparse
from itertools import chain

components = 'components'

# In[1]:

parser = argparse.ArgumentParser(description = 'Synthetic Image Cutout Generator')

parser.add_argument('-n', '--num_objects', type=int, help='Maximum number of objects per image.')
parser.add_argument('-d', '--dimensions', type=int, help='Dimensions of images (256, 512, ...)')
parser.add_argument('-i', '--image_number', type=int, help='[REQUIRED] Number of training set images to generate.')
parser.add_argument('-v', '--val_number', type=int, help='Number of validation set images to generate.')
parser.add_argument('-t', '--test_number', type=int, help='Number of test set images to generate.')
parser.add_argument('-o', '--output', type=str, help='Directory for images to be stored (default = output).')

args = parser.parse_args()

bg_dimension = args.dimensions

if args.dimensions == None:
    bg_dimension = 1024

folder = f'{args.output}'

if args.output == None:
    folder = 'output/' 

max_objs = args.num_objects

if args.num_objects == None:
    max_objs = 10

number = args.image_number
    
if args.image_number == None:
    sys.exit('Please add -i argument for the number of training set images you would like to create')

n_val = args.val_number
    
if args.val_number == None:
    n_val = int(number/10)

n_test = args.test_number    

if args.test_number == None:
    n_test = int(number/10)

#Directory for components
img_files = os.listdir(os.path.join(components, 'images'))
bg_files = os.listdir(os.path.join(components, 'bg'))

for c in reversed(range(len(bg_files))):
    if not bg_files[c].endswith(('.png', '.jpg', '.tif')):
        del bg_files[c]
        
for c in reversed(range(len(img_files))):
    if not img_files[c].endswith(('.png', '.jpg', '.tif')):
        del img_files[c]
        
        
def generate_dataset(number, start, split):

    img_ann = []

    ann = []

    # In[2]:
    for h in range(0, number):
        
        i = h + start
        '''
        # (Optional) MORE COMPUTATIONALLY INTENSIVE: Randomly select bg for each image generated
        bg_file = random.choice(bg_files)
        bg = Image.open(os.path.join(components, 'bg/{}'.format(bg_file)))
        width, height = bg.size
        x = random.randint(0, width - bg_dimension)
        y = random.randint(0, height - bg_dimension)
        image = bg.crop((x, y, x + bg_dimension, y + bg_dimension))
        '''
        
        bg_file = random.choice(bg_files)
        try:
            #if bg image is corrupted, skips and makes blank white bg
            image  = Image.open(os.path.join(components, 'bg/{}'.format(bg_file)))
            image = image.convert("RGBA")
        except:
            image = Image.new('RGBA', (bg_dimension, bg_dimension), color='white')
        objs = np.random.randint(max_objs)
        
        img_ann.append({"id": i, "width": bg_dimension, "height": bg_dimension, "file_name": "{}".format(i)+".png"})
        
        j = 0
        #makes each object id unique (required for coco format)
        obj_start = i*max_objs

        while j <= objs:
            
            file = random.choice(img_files)
            im = Image.open(os.path.join(components, 'images/{}'.format(file)))
            im = im.convert("RGBA")
            mk = Image.open(os.path.join(components, 'masks/{}'.format(file)))
            mk = mk.convert("L")
            wt, ht = im.size
            
            angle = random.randint(-180, 180)
            im = im.rotate(angle, expand=True)
            mk = mk.rotate(angle, expand=True)
            
            # randomly resize object (can be altered but will create more image stretching/compression)
            randw = round(random.uniform(0.8, 1.2), 2)
            randh = round(random.uniform(0.8, 1.2), 2)
            wid = wt*randw
            hei = ht*randh

            dim = (int(wid), int(hei))

            im = im.resize(dim)
            mk = mk.resize(dim)
            
            w, h = im.size
            
            if w < bg_dimension or h < bg_dimension:
            
                w = max(bg_dimension, w)
                h = max(bg_dimension, h)

                ibk = Image.new("RGBA", (w, h))
                mbk = Image.new("L", (w, h))
                ibk.paste(im, (0,0))
                mbk.paste(mk, (0,0))
                im = ibk
                mk = mbk

            x = random.randint(0, int(w - bg_dimension))
            y = random.randint(0, int(h - bg_dimension))
            cutout = im.crop((x, y, x + bg_dimension, y + bg_dimension))
            mask = mk.crop((x, y, x + bg_dimension, y + bg_dimension))

            if random.random() < 0.25:
                cutout = cutout.transpose(method=Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)

            if random.random() < 0.25:
                cutout = cutout.transpose(method=Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(method=Image.FLIP_TOP_BOTTOM)

            mkarr = np.array(mask)
            pix = np.sum(mkarr >= 1)

            if pix > 50: #added so that too small corners of hair are ignored, improves accuracy of object vs debris
                
                cut = Image.new("RGBA", image.size, (255,255,255,255))
                cut.paste(cutout, (0, 0), mask)
                #adds transparency to image to mimic real images more closely
                tv = np.random.randint(180, 255)
                alp = mask.point(lambda i: tv if i>127 else 0)
                cut.putalpha(alp)
                pixels = cut.getdata()
                
                alpha = []
                for pixel in pixels:
                    if pixel[0] >= 225 and pixel[1] >= 225 and pixel[2] >= 225 and pixel[3] > 127:
                        alpha.append((pixel[0], pixel[1], pixel[2], 0))
                    else:
                        alpha.append(pixel)
                
                cut.putdata(alpha)
    
                image = Image.alpha_composite(image, cut)

                tempmask = cv2.cvtColor(mkarr, cv2.COLOR_GRAY2BGR)
                imgray = cv2.cvtColor(tempmask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=cv2.contourArea)

                #coco-style format
                ann.append({"iscrowd": 0, "id": (obj_start + j), "image_id": i, "category_id": 1, "bbox": cv2.boundingRect(contour), "area": cv2.contourArea(contour), "segmentation": [contour.flatten().tolist()]})

                j += 1
                
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #randomize blur added to image after all objects are added
        blr = np.random.randint(1, 5)
        blur_type = random.choice(["blur", "bilat"]) #removed gauss blur
        if blur_type == "blur":
            image = cv2.blur(image, (blr, blr))
        #elif blur_type == "gauss":
        #    image = cv2.GaussianBlur(image, (blr, blr), 0)
        elif blur_type == "bilat":
            image = cv2.bilateralFilter	(image, blr, 50, 50)
            
        if random.random() < 0.25:
            # create an image with a single color
            rand_filter = np.full((bg_dimension,bg_dimension,3), (random.randint(0,255),random.randint(0,255),random.randint(0,255)), np.uint8)

            # add color filter
            image  = cv2.addWeighted(image, 0.9, rand_filter, 0.1, 0)

        cv2.imwrite(os.path.join(folder, split, "images/{}".format(i)+".png"), image)
    
    # allows for multithreading while saving to single coco-format annotations file
    output_lockann.acquire()
    try:    
        output_ann.put(ann)
        output_ann.task_done()
    finally:
        output_lockann.release()

    output_ann.join()

    output_lockimg.acquire()
    try:
        output_img.put(img_ann)
        output_img.task_done()
    finally:
        output_lockimg.release()

    output_img.join()
        
#defines task for each thread
def generate_chunk(start, end, split):
    print(f'generating dataset {start}')
    generate_dataset(int(end-start), start, split)
    pass

#allocates task to each thread
def generate_dataset_parallel(num_items, chunk_size, split):
    
    if not os.path.exists(os.path.join(folder, split, 'images')):
        os.makedirs(os.path.join(folder, split, 'images'))

    if not os.path.exists(os.path.join(folder, split, 'labels')):
        os.makedirs(os.path.join(folder, split, 'labels'))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            futures.append(executor.submit(generate_chunk, start, end, split))
        concurrent.futures.wait(futures)
        
    #reformats all annotations into coco format and saves file (can be modified to support more than 1 object type)
    coco_format = {"images": [{}], "categories": [{}], "annotations": [{}]}

    ccat = []
    cimg = []
    cann = []
    
    while ((output_img.empty())!=True):
        for z in output_img.get():
            cimg.append(z)
    
    while ((output_ann.empty())!=True):
        for z in output_ann.get():
            cann.append(z)

    coco_format["categories"] = [{"id": 1, "name": 'hair', "supercategory": 'hair'}]
    coco_format["images"] = cimg
    coco_format["annotations"] = cann


    labpath = os.path.join(folder, split, 'labels')

    if not os.path.exists(labpath):
        os.makedirs(labpath) 
    
    with open((os.path.join(labpath, 'coco.json')), "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)

        
output_img = queue.Queue()
output_ann = queue.Queue()

output_lockimg = threading.Lock()
output_lockann = threading.Lock()
        
#generates training, test, and validation datasets
generate_dataset_parallel(number, 1, split = 'train')
generate_dataset_parallel(n_val, 1, split = 'val')
generate_dataset_parallel(n_test, 1, split = 'test')