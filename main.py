import logging
from pathlib import Path
import os
import torch as t
import yaml
from pycocotools.cocoeval import COCOeval
# from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import model
from torchvision import transforms
import quan as q
import munch
import json
import numpy as np
import time
import csv
# import cv2

logger = logging.getLogger()

def get_config(default_file):
    # p = argparse.ArgumentParser(description='Learned Step Size Quantization')
    # p.add_argument('--config_file', metavar='PATH', nargs='+', default=str(default_file),
    #                help='path to a configuration file')
    # arg = p.parse_args()

    with open(default_file, encoding='utf-8') as yaml_file:
        cfg = yaml.safe_load(yaml_file)  # 以yaml格式加载成字典实例

    # for f in arg.config_file:
    #     if not os.path.isfile(f):
    #         raise FileNotFoundError('Cannot find a configuration file at', f)
    #     with open(f) as yaml_file:
    #         c = yaml.safe_load(yaml_file)
    #         cfg = merge_nested_dict(cfg, c)

    return munch.munchify(cfg)

def evaluate_coco(dataset, model, threshold=0.05):
    
    model.eval()
    
    with t.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if t.cuda.is_available():
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        model.train()

        return


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)



def detect_image(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = t.load(model_path)

    if t.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with t.no_grad():

            image = t.from_numpy(image)
            if t.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imshow('detections', image_orig)
            cv2.waitKey(0)

def save_checkpoint( model,arch = 'retinanet' ,extras=None, is_best=None, name=None, output_dir='/ckpt'):
    """Save a pyTorch training checkpoint
    Args:
        epoch: current epoch number
        arch: name of the network architecture/topology
        model: a pyTorch model
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        output_dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(output_dir):
        raise IOError('Checkpoint directory does not exist at', os.path.abspath(dir))

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')

    filename = 'checkpoint.pt' if name is None else name + '_checkpoint.pt'
    filepath = os.path.join(output_dir, filename)
    filename_best = 'best.pt' if name is None else name + '_best.pt'
    filepath_best = os.path.join(output_dir, filename_best)

    checkpoint = {
        'state_dict': model.state_dict(),
        'arch': arch,
        'extras': extras,
    }

    msg = 'Saving checkpoint to:\n'
    msg += '             Current: %s\n' % filepath
    t.save(checkpoint, filepath)
    if is_best:
        msg += '                Best: %s\n' % filepath_best
        t.save(checkpoint, filepath_best)
    logger.info(msg)


def main():
    retinanet_model = model.resnet50(num_classes=80,)
    retinanet_model.load_state_dict(t.load('retinanet.pt',map_location=lambda storage, loc: storage),strict=False)
    print(type(retinanet_model))
    # dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    script_dir = Path.cwd()
    print(script_dir)
    args = get_config(default_file=script_dir/'config.yaml')
    print(args)
    modules_to_replace, ori_modules, names_to_replace = q.find_modules_to_quantize(retinanet_model, args.quan)
    print(modules_to_replace)
    new_model = q.replace_module_by_names(retinanet_model,modules_to_replace)
    t.save(new_model.state_dict(),'quan_model0.pt',_use_new_zipfile_serialization=False)
    model2 = model.resnet50(num_classes=80,)
    model2.load_state_dict(t.load('quan_model0.pt',map_location=lambda storage, loc: storage),strict=False)
    print(model2)
    # save_checkpoint(new_model,output_dir=script_dir/'ckpt')
    # print(new_model)
    # detect_image('test_0.jpg','quan_model.pt')




if __name__ == "__main__":
    main()
