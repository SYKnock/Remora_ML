import os
import argparse
import cv2
import numpy as np
import functools as func
from PIL import Image
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable

from dtr.model import Model as Model
from dtr.utils import AttnLabelConverter as AttnLabelConverter
from dtr.dataset import AlignCollate as AlignCollate
from dtr.dataset import ResizeNormalize as ResizeNormalize

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
    
def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    
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

def predict(opt, imgs):
    opt.batch_size = len(imgs)
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load('dtr/weights/TPS-ResNet-BiLSTM-Attn.pth', map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=False)
    
    # predict
    model.eval()
    texts = []
    
    with torch.no_grad():
        transform = ResizeNormalize((opt.imgW, opt.imgH))
        image_tensors = [transform(Image.fromarray(image).convert('L')) for image in imgs]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        for pred in preds_str:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            texts.append(pred)
    
    return texts


root = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt = argparse.Namespace(
        Transformation='TPS',
        num_fiducial=20,
        imgH=32,
        imgW=100,
        input_channel=1,
        output_channel=512,
        hidden_size=256,
        FeatureExtraction='ResNet',
        SequenceModeling='BiLSTM',
        num_class=0,
        batch_max_length=25,
        rgb=False,
        character='0123456789abcdefghijklmnopqrstuvwxyz',
        Prediction='Attn'
    )

net = CRAFT()
net.load_state_dict(copyStateDict(torch.load(root+'/CRAFT/weights/craft_mlt_25k.pth', map_location=device)))
net = net.cuda()

net = torch.nn.DataParallel(net)
cudnn.benchmark = False

net.eval()

f = open(root+'/predict_text.txt','w')
f.close()

image_list = os.listdir(root+'/frames')
for file in image_list:
    file_name, file_extension = os.path.splitext(file)
    if file_extension != '.png':
        image_list.remove(file)
        

image_list = sorted(image_list,key=lambda x: int(x[5:-4]))

thredhold = 13
pred_text = []

for image_path in image_list:
    try:
        image = imgproc.loadImage(root+'/frames/'+image_path)
    except:
        continue
    if not(type(image) is np.ndarray):
        continue
    
    boxes, polys = test_net(net=net,image=image,text_threshold=0.7,link_threshold=0.4,low_text=0.4,cuda=True,poly=False)

    pos = []

    for poly in polys:
        x1 = int(min(poly[:, 0]))
        y1 = int(min(poly[:, 1]))
        x2 = int(max(poly[:, 0]))
        y2 = int(max(poly[:, 1]))

        if y2-y1>=thredhold:
            pos.append([x1, y1, x2, y2])

    height = image.shape[0]
    width = image.shape[1]
    
    # 이미지의 하단에 있는 박스만 남김
    pos = [item for item in pos if item[1] > (height * 0.7) and item[1] < (height * 0.95) and item[0] > (0.15 * width)]
    # 박스를 왼쪽에서 오른쪽으로, 위에서 아래로 정렬
    sorted_pos = sorted(pos, key=func.cmp_to_key(compare))

    if len(sorted_pos)==0:
        continue

    imgs = []

    for p in sorted_pos:
        img = image[p[1]:p[3],p[0]:p[2]]
        imgs.append(img)

    pred_text.append(predict(opt,imgs))

sentences = []
for texts in pred_text:
    s = ""
    for tmp in texts:
        s += tmp + ' '

    s = s[:-1] + '\n'
    sentences.append(s)

pre = -1000
texts = []
text = np.array([])
scores = np.array([])

for s in sentences:
    s_np = np.array(list(s))
    len_list = len(s_np)-list(s_np).count(' ')-list(s_np).count('\n')
    score = (np.frombuffer(s_np,'uint8').sum()-(ord('a')*len_list))

    if (abs(score - pre) > 150):
        if pre !=-1000:
            texts.append(text[np.argmin(abs(scores-scores.mean()))])
        text = np.array([s])
        scores = np.array([score])
        pre = score
    else:
        text = np.append(text,s)
        scores = np.append(scores,score)

texts.append(text[np.argmin(abs(scores-scores.mean()))])

with open(root+'/predict_text.txt','w') as f:
    for text in texts:
        f.write(text)

f.close()