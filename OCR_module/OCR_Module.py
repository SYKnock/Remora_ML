import OCR_Detection_KOR as detect
import OCR_Recognition_KOR as recognize
import os
import shutil
root = os.getcwd()

if __name__ == "__main__":
    detect.detection('/frames')
    script = recognize.recognition('/detection_result')
    txt_file = open("ocr_result.txt", "w")
    # txt_file_for_classification = open("../Classification_module/classification_input.txt")
    txt_file.write(script)
    # txt_file_for_classification.write(script)
    txt_file.close()
    # txt_file_for_classification.close()

    frame_deletion_list = os.listdir(root + '/frames')
    for file in frame_deletion_list:
        os.remove(root + '/frames/' + file)
    
    detection_deletion_list = os.listdir(root + '/detection_result')
    for dir in detection_deletion_list:
        shutil.rmtree(root + '/detection_result/' + dir)

