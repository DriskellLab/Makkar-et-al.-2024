Synthetic Data Generation

Requires: gen_data.py, components folder (with images, masks, and bg subdirectories)

gen_data.py:

    Run with "python gen_data.py"

    Additional arguments:
        -i (REQUIRED): Number of training set images to generate.
        -n : Maximum number of objects per image. (default = 10)
        -d : Dimensions of generated images (256, 512, ...). (default = 1024)
        -v : Number of validation set images to generate. (default = 10% of number of training images)
        -t : Number of test set images to generate. (default = 10% of number of training images)
        -o : Directory for images to be stored. (default = output)

components folder:

    images:
        Sample images of objects of interest (i.e. hair fibers).
    
    masks:
        Binary masks of annotated sample images (must be labelled with same name as sample image!)
    
    bg:
        Sample background images (must be same size as desired dimensions for generated images).
    
    (OPTIONAL) bg_cutout.py:
        Allows for the creation of sample background images. Type in file name of large background (i.e. background.png) and enter dimensions and number of bg images to generate.