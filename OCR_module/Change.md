change CRAFT-pytorch directory name as CRAFT  
change deep-text-recognition-benchmark directory name as dtr  

in CRAFT/craft.py: from basenet.vgg16_bn -> from CRAFT.basenet.vgg16_bn  
in CARFT/file_utils.py: import imgproc -> import CRAFT.imgproc  
in dtr/model.py: write 'dtr' in front of every "models." (example: from dtr.models.transformation ...)  