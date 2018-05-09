#!/bin/bash

# switch to the environment: tensorflow
source activate tensorflow

# change permission of the ttyUSB0
sudo chmod 777 /dev/ttyUSB0

# initialize the lens
python test_usb.py

# run the program
# python demo_data_collection_offset.py
python pulsecam_iccv9.py
# python pulsecam_ext.py
#python get_hdr_psfs.py
