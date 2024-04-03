Process Images

Requires: tilescan.py, config folder, models folder

tilescan.py:

    Run with "python tilescan.py"

    Additional arguments:
        -i: Directory for images to run inference on (default = input)
        -m: Directory for model/weights file to run inference with (default = models/epoch_1000.pth)
        -c: Directory for config file to run inference with (default = config/mrcnn_r152.py)
        -o: Directory for image cutouts to be stored (default = output)
        -r: Directory for quantifications to be stored (default = results)
        -s: Size of tiles to process (default = 1024)
        -p: Percent overlap of tiles for processing (default = 0.25)

config folder:

    mrcnn_r152.py: configuration file for the current implementation of the model (utilizes resnet152).
    mrcnn_r101: (optional) alternative configuration file for an implementation that uses resnet101.
    mrcnn.py: script that defines the training data location and batch sizes for training.
    schedule.py: script that defines the duration of training.
    default_runtime.py: script that holds configuration preferences (i.e. hook: method of saving model training statistics).
    
models folder:
    
    epoch_1000.pth: trained model for detection of 1024x1024 cutouts of murine hair fibers.