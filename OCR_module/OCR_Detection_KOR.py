import os
import numpy as np
import functools as func

import torch

import torch.backends.cudnn as cudnn
from torch.autograd import Variable


from collections import OrderedDict

import cv2
import numpy as np
import CRAFT.craft_utils as craft_utils
import CRAFT.imgproc as imgproc
import CRAFT.file_utils as file_utils
from CRAFT.craft import CRAFT as CRAFT


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def compare(x, y):
    check_line = abs(x[1] - y[1])
    if check_line < 15:
        return x[0] - y[0]
    else:
        return x[1] - y[1]


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)   
    x = Variable(x.unsqueeze(0))               
    
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()
    
    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys


def detection(path):
    root = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_list = os.listdir(root + path)
    dest = './detection_result/'

    if not os.path.isdir(dest):
        os.mkdir(dest)
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(root + '/CRAFT/weights/craft_mlt_25k.pth', map_location=device)))
   
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

    net.eval()

    for file in image_list:
        file_name, file_extension = os.path.splitext(file)
        if file_extension != '.png':
            image_list.remove(file)

    image_list = sorted(image_list, key=lambda x: int(x[2:-4]))
   
    for image in image_list:
        img_path = root + path + '/' + image
        target = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = target.shape[:2]
        crop = target[int(height * 0.7): int(height * 0.93), int(width * 0.1):int(width * 0.95)]
        cv2.imwrite(img_path, crop)

    thredhold = 13

    cnt = 0

    for image_path in image_list:
        try:
            image = imgproc.loadImage(root + path + '/' + image_path)
        except:
            continue
        if not(type(image) is np.ndarray):
            continue
        
        boxes, polys = test_net(net=net, image=image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, cuda=True, poly=False)

        pos = []

        for poly in polys:
            x1 = int(min(poly[:, 0]))
            y1 = int(min(poly[:, 1]))
            x2 = int(max(poly[:, 0]))
            y2 = int(max(poly[:, 1]))

            if y2 - y1 >= thredhold:
                pos.append([x1, y1, x2, y2])

        # 박스를 왼쪽에서 오른쪽으로, 위에서 아래로 정렬
        sorted_pos = sorted(pos, key=func.cmp_to_key(compare))

        if len(sorted_pos) == 0:
            continue

        dir_path = root + "/detection_result/" + str(cnt)
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except OSError:
            print('Error: Creating Directory, ' + dir_path)

        img_cnt = 0

        for p in sorted_pos:
            img = image[p[1]:p[3], p[0]:p[2]]
            img_path = dir_path + "/" + str(img_cnt) + ".png"

            if img.size != 0:
                cv2.imwrite(img_path, img)
            img_cnt += 1
        cnt += 1
