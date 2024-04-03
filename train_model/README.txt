Train Model

Requires: OBBDetection, config folder

OBBDetection:

    Install OBBDetection as found at: https://github.com/jbwang1997/OBBDetection

config folder (for training coco-style formatted data):

    mrcnn_r152.py: configuration file for the current implementation of the model (utilizes resnet152).
    mrcnn_r101: (optional) alternative configuration file for an implementation that uses resnet101.
    mrcnn.py: script that defines the training data location and batch sizes for training.
    schedule.py: script that defines the duration of training.
    default_runtime.py: script that holds configuration preferences (i.e. hook: method of saving model training statistics).