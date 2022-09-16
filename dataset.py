import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.data import DataLoader
import pyclipper
import Polygon as plg
from shapely.geometry import Polygon

### for data augmentation ###
def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=640):
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    img = scale_aligned(img, scale)
    return img


def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[3] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[3] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, 512 - t_h, 0, 512 - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, 512 - t_h, 0, 512 - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask
### for data augmentation ###



def my_transforms():
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform

def get_anno(img, gt_path):
    h, w = img.shape[0:2]
    bboxes = []
    f1 = open(gt_path, 'r')
    lines = f1.readlines()
    # import pdb;pdb.set_trace()
    for line in lines[:]:
        line = line.strip().split(',')
        # import pdb;pdb.set_trace()
        bbox = []
        for i in range(len(line)):
            bbox.append(float(line[i]))
        point_num = int(len(line)/2)
        # import pdb;pdb.set_trace()
        bbox = np.asarray(bbox)/ ([w * 1.0, h * 1.0] * point_num)
        bboxes.append(bbox)
    return bboxes

def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes

def draw_border_map(polygon, canvas, mask_ori, mask):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    ### shrink box ###
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * \
        (1 - np.power(0.95, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(-distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
    ### shrink box ###

    cv2.fillPoly(mask_ori, [polygon.astype(np.int32)], 1.0)

    polygon = padded_polygon
    polygon_shape = Polygon(padded_polygon)
    distance = polygon_shape.area * \
        (1 - np.power(0.4, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros(
        (polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        # import pdb;pdb.set_trace()
        absolute_distance = coumpute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = distance_map.min(axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid-ymin:ymax_valid-ymax+height,
            xmin_valid-xmin:xmax_valid-xmax+width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

def coumpute_distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(
        xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(
        xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(
        point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
        (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 *
                     square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(
        square_distance_1, square_distance_2))[cosin < 0]
    # extend_line(point_1, point_2, result)
    return result

def get_seg_map(img, label):
    canvas = np.zeros(img.shape[:2], dtype = np.float32)
    mask = np.zeros(img.shape[:2], dtype = np.float32)
    mask_ori = np.zeros(img.shape[:2], dtype = np.float32)
    polygons = label

    for i in range(len(polygons)):
        draw_border_map(polygons[i], canvas, mask_ori, mask=mask)
    return canvas, mask, mask_ori


class Dataset(torch.utils.data.Dataset):
    def __init__(self, flist, training, input_size):
        super(Dataset, self).__init__()
        self.training = training
        self.data = self.load_flist(flist)

        self.input_size = input_size

    def __len__(self):
        return len(self.data)
        # return 100

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = cv2.imread(self.data[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        name = self.data[index]

        gt = cv2.imread(self.data[index].replace('all_images', 'all_labels'))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        ### structure ###
        structure_im = cv2.imread(self.data[index].replace('all_images', 'structure_im'))
        structure_im = cv2.cvtColor(structure_im, cv2.COLOR_BGR2RGB)

        if self.training:
            structure_lbl = cv2.imread(self.data[index].replace('all_images', 'structure_lbl'))
            structure_lbl = cv2.cvtColor(structure_lbl, cv2.COLOR_BGR2RGB)
        else:
            structure_lbl = structure_im
        ### structure ###

        gt_text, soft_mask = self.load_detection_anno(img, index)

        if self.training:
            imgs = [img, gt, gt_text, soft_mask, structure_im, structure_lbl]
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs) 
            img, gt, gt_text, soft_mask, structure_im, structure_lbl = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5] 

        img = self.to_tensor(img)
        gt = self.to_tensor(gt)

        structure_im = self.to_tensor(structure_im)
        structure_lbl = self.to_tensor(structure_lbl)

        gt_text = torch.from_numpy(gt_text).long()
        soft_mask = torch.from_numpy(soft_mask)

        return img, gt, structure_im, structure_lbl, gt_text, soft_mask, index, name

    ### for detection ###
    def load_detection_anno(self, img, index):
        if self.training:
            gt_path = self.data[index].replace('all_images', 'all_gts').replace('jpg', 'txt')
            bboxes = get_anno(img, gt_path)

        ##################### test #####################
        else:
            gt_path = self.data[index].replace('all_images', 'all_gts').replace('jpg', 'txt')
            bboxes = get_anno(img, gt_path)
        ##################### test #####################

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')

        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] //2)),
                                        (bboxes[i].shape[0] // 2, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)

        gt_text= gt_instance.copy()
        gt_text[gt_text > 0] = 1
  
        canvas, shrink_mask, mask_ori = get_seg_map(img, bboxes)
        soft_mask = canvas + mask_ori
        index_mask = np.where(soft_mask > 1)
        soft_mask[index_mask] = 1

        return gt_text, soft_mask
    ### for detection ###

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                # print(np.genfromtxt(flist, dtype=np.str))
                # return np.genfromtxt(flist, dtype=np.str)
                try:
                    return np.genfromtxt(flist, dtype=np.str)
                except:
                    return [flist]
        return []

def build_dataloader(flist, training, input_size, batch_size, num_workers, shuffle):

    dataset = Dataset(
        flist=flist,
        training=training,
        input_size=input_size
        )
    print('Total instance number:', dataset.__len__())

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle
    )

    return dataloader

