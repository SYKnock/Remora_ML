import OCR_Detection_KOR as detect
import OCR_Recognition_KOR as recognize
import os
import shutil
root = os.getcwd()

if __name__ == "__main__":
    detect.detection('/frames')
    script = recognize.recognition('/detection_result')
    txt_file = open("predict.txt", "w")
    txt_file.write(script)
    txt_file.close()

    frame_deletion_list = os.listdir(root + '/frames')
    for file in frame_deletion_list:
        os.remove(root + '/frames/' + file)
    
    detection_deletion_list = os.listdir(root + '/detection_result')
    for dir in detection_deletion_list:
        shutil.rmtree(root + '/detection_result/' + dir)

