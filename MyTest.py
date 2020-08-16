from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import sys
from utils import *
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer

parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('-m', '--trained_model', default='weights/Final_FaceBoxes.pth',type=str, help='Trained state_dict file path to open')
parser.add_argument('--video', type=str, default='',help='path to video file')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--src', type=int, default=0,help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',help='path to the output directory')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))



def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)


    resize=2.5
    _t = {'forward_pass': Timer(), 'misc': Timer()}

    #testing begin
    wind_name = 'face detection in surviellance camera using FaceBoxes'
    output_file = ''

    if args.video:
        if not os.path.isfile(args.video):
            print("[!] ==> Input video file {} doesn't exist".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4].rsplit('/')[-1] + '_Facebox.avi'
    else:
        # Get data from the camera
        cap = cv2.VideoCapture(args.src)
        output_file = args.video[:-4].rsplit('/')[-1] + '_webcamFaceBoxV.avi'




    # Get the video writer initialized to save the output video
    video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   cap.get(cv2.CAP_PROP_FPS), (
                                       round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:

        has_frame, img = cap.read(cv2.IMREAD_COLOR)

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break
        else:
           frame = np.float32(img)
           if resize != 1:
               frame= cv2.resize(frame, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
           IMG_WIDTH, IMG_HEIGHT, _ = frame.shape
           # Create a 4D blob from a frame.
           #blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)
           # Sets the input to the network
           #net.setInput(blob)
           # Runs the forward pass to get output of the output layers
           scale = torch.Tensor([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
           frame -= (104, 117, 123)
           frame = frame.transpose(2, 0, 1)
           frame= torch.from_numpy(frame).unsqueeze(0)
           frame = frame.to(device)
           scale = scale.to(device)

           _t['forward_pass'].tic()
           loc, conf = net(frame)  # forward pass
           _t['forward_pass'].toc()
           _t['misc'].tic()
           priorbox = PriorBox(cfg, image_size=(IMG_WIDTH, IMG_HEIGHT))
           priors = priorbox.forward()
           priors = priors.to(device)
           prior_data = priors.data
           boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
           boxes = boxes * scale / resize
           boxes = boxes.cpu().numpy()
           scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
           # Remove the bounding boxes with low confidence
           # faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
           inds = np.where(scores > args.confidence_threshold)[0]
           boxes = boxes[inds]
           scores = scores[inds]
           # keep top-K before NMS
           order = scores.argsort()[::-1][:args.top_k]
           boxes = boxes[order]
           scores = scores[order]

           # do NMS
           faces=np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
           keep = nms(faces, args.nms_threshold, force_cpu=args.cpu)
           faces = faces[keep, :]
           # keep top-K faster NMS
           faces= faces[:args.keep_top_k, :]
           _t['misc'].toc()
            #draw bondingBox

           if len(faces)!=0:
             for b in faces:
               if b[4] < args.vis_thres:
                  continue
               text = "face"+"{:.4f}".format(b[4])
               b = list(map(int, b))
               cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0,255 , 0), 2)
               cx = b[0]
               cy = b[1] + 12
               cv2.putText(img, text, (cx, cy),
                          cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
           #show the vid
           cv2.imshow(wind_name, img)
           # save the output video
           video_writer.write(img.astype(np.uint8))
           key = cv2.waitKey(1)
           if key == 27 or key == ord('q'):
                print('[i] ==> Interrupted by user!')
                break

           facecor=faces
           faces=[]

    cap.release()
    cv2.destroyAllWindows()

    print(facecor)
    print('==> All done!')
    print('***********************************************************')