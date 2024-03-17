#!/usr/bin/env python
from scipy.spatial import distance as dist
import imutils
from imutils.video import VideoStream, FPS
from imutils import face_utils
from threading import Thread
import net
import time
import shutil
import argparse
import time
import os
import cv2
import pyshine as ps
import torch
from typing import Dict
import json
import urllib
import itertools
import logging
import os
import argparse
import torchmetrics
import pytorchvideo.data
import pytorchvideo.models
import pytorchvideo.models.resnet
import pytorch_lightning
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
"""
class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args):
        self.args = args
        super().__init__()
        #===============this cause error=================
        #self.train_accuracy = pytorch_lightning.metrics.Accuracy()
        #self.val_accuracy = pytorch_lightning.metrics.Accuracy()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        if self.args.arch == "video_resnet":
            self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3,
                model_num_class=2,
            )
            self.batch_key = "video"
        elif self.args.arch == "video_slowfast":
            self.model = pytorchvideo.models.slowfast.create_slowfast(
                #input_channels=3,
                model_num_class=2,
            )
            self.batch_key = "video"
        elif self.args.arch == "video_vgg16":
            self.model = net.ConvLstm(768, 256, 2, True, 2)
            self.batch_key = "video"
        elif self.args.arch == "audio_resnet":
            self.model = pytorchvideo.models.resnet.create_acoustic_resnet(
                input_channel=1,
                model_num_class=2,
            )
            self.batch_key = "audio"
        else:
            raise Exception("{self.args.arch} not supported")

    def on_train_epoch_start(self):

        epoch = self.trainer.current_epoch

    def forward(self, x):
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss)
        #print('batch is',batch_idx)
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_idx
        )

        return loss


    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["train_loss"] for x in training_step_outputs]).mean()
        self.logger.experiment.add_scalar('loss',avg_loss, self.current_epoch)
  
        output_file = './result.txt'
        if os.path.exists(output_file):
            output_file = open(output_file, 'a')
        else :
            output_file = open(output_file, 'w')
            print("Create result.txt.")             
        output_file.write('epoch={}, Train loss {:.8f}, Train acc {:.3f}'.format( self.current_epoch, loss, acc))
        output_file.close()

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle. For this
        simple example it's mostly the same as the training loop but with a different
        metric name.
        """
        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("val_loss", loss)
        self.log(
            "val_acc", acc, on_step=False, batch_size=batch_idx, prog_bar=True, sync_dist=True
        )

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar('loss',avg_loss, self.current_epoch)
        output_file = './result.txt'
        if os.path.exists(output_file):
            output_file = open(output_file, 'a')
        else :
            output_file = open(output_file, 'w')
            print("Create result.txt.")             
        output_file.write('epoch={}, Val loss {:.8f}, Val acc {:.3f}\n'.format( self.current_epoch, avg_loss, acc))
        output_file.close()

    def configure_optimizers(self):
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]

parser = argparse.ArgumentParser()
parser.add_argument("--on_cluster", action="store_true")
parser.add_argument("--job_name", default="ptv_video_classification", type=str)
parser.add_argument("--working_directory", default=".", type=str)
parser.add_argument("--partition", default="dev", type=str)
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument(
    "--arch",
    default="video_resnet",
    choices=["video_resnet", "audio_resnet","video_slowfast", "video_vgg16"],
    type=str,
)
parser.add_argument("--data_path", default='data', type=str)
parser.add_argument("--video_path_prefix", default="", type=str)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--clip_duration", default=2, type=float)
parser.add_argument(
    "--data_type", default="video", choices=["video", "audio"], type=str
)
parser.add_argument("--video_num_subsampled", default=8, type=int)
parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
parser.add_argument("--video_crop_size", default=224, type=int)
parser.add_argument("--video_min_short_side_scale", default=256, type=int)
parser.add_argument("--video_max_short_side_scale", default=320, type=int)
parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)
parser.add_argument("--audio_raw_sample_rate", default=44100, type=int)
parser.add_argument("--audio_resampled_rate", default=16000, type=int)
parser.add_argument("--audio_mel_window_size", default=32, type=int)
parser.add_argument("--audio_mel_step_size", default=16, type=int)
parser.add_argument("--audio_num_mels", default=80, type=int)
parser.add_argument("--audio_mel_num_subsample", default=128, type=int)
parser.add_argument("--audio_logmel_mean", default=-7.03, type=float)
parser.add_argument("--audio_logmel_std", default=4.66, type=float)
parser.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system")
parser = pytorch_lightning.Trainer.add_argparse_args(parser)
parser.add_argument("--video_name", default="sleepyCombibation.avi", type=str)
parser.set_defaults(
    max_epochs=200,
    callbacks=[LearningRateMonitor()],
    replace_sampler_ddp=False,
)
class LitMCdropoutModel(pytorch_lightning.LightningModule):
    def __init__(self, model, mc_iteration):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout()
        self.mc_iteration = mc_iteration

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        self.dropout.train()

        # take average of `self.mc_iteration` iterations
        pred = torch.vstack([self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]).mean(dim=0)
        return pred

class KineticsDataModule(pytorch_lightning.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

    def _make_transforms(self):

        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]
        elif self.args.data_type == "audio":
            transform = [
                self._audio_transform(),
                RemoveKey("video"),
            ]

        return Compose(transform)

    def _video_transform(self):

        args = self.args
        if self.args.arch == "video_slowfast": 
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            sampling_rate = 2
            frames_per_second = 30
            alpha = 4
            return  ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames),
                            Lambda(lambda x: x/255.0),
                            NormalizeVideo(mean, std),
                            ShortSideScale(
                                size=side_size
                            ),
                            CenterCropVideo(crop_size),
                            PackPathway()
                        ]
                    ),
                )
        else:
            return ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(args.video_num_subsampled),
                        Normalize(args.video_means, args.video_stds),
                    ]
                    + (
                        [
                            RandomShortSideScale(
                                min_size=args.video_min_short_side_scale,
                                max_size=args.video_max_short_side_scale,
                            ),
                            RandomCrop(args.video_crop_size),
                            RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                        ]
                        if mode == "train"
                        else [
                            ShortSideScale(args.video_min_short_side_scale),
                            CenterCrop(args.video_crop_size),
                        ]
                    )
                ),
            )

    def train_dataloader(self):

        sampler = RandomSampler
        train_transform = self._make_transforms(mode="train")
        self.train_dataset = LimitDataset(
            pytorchvideo.data.Kinetics(
                data_path=os.path.join(self.args.data_path, "train.csv"),
                clip_sampler=pytorchvideo.data.make_clip_sampler(
                    "random", self.args.clip_duration
                ),
                video_path_prefix=self.args.video_path_prefix,
                transform=train_transform,
                video_sampler=sampler,
            )
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):

        sampler = RandomSampler
        val_transform = self._make_transforms(mode="val")
        self.val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.args.data_path, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.args.clip_duration
            ),
            video_path_prefix=self.args.video_path_prefix,
            transform=val_transform,
            video_sampler=sampler,
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

class LimitDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos

class PackPathway(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, frames):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

args = parser.parse_args()
path = "./best_model_final.ckpt"
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = VideoClassificationLightningModule.load_from_checkpoint( path, map_location='cpu',args=args )
model = model.to(device)
model.eval()

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(args.video_num_subsampled),
            Normalize(args.video_means, args.video_stds),
        ]
        + (
            [
                RandomShortSideScale(
                    min_size=args.video_min_short_side_scale,
                    max_size=args.video_max_short_side_scale,
                ),
                RandomCrop(args.video_crop_size),
                RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                ShortSideScale(args.video_min_short_side_scale),
                CenterCrop(args.video_crop_size),
            ]
        )
    ),
)
"""
clip_sampler=pytorchvideo.data.make_clip_sampler("random", args.clip_duration)
clip_start_sec = 0.0 # secs
clip_duration = (num_frames * sampling_rate)/frames_per_second
"""

clip_start_sec = 0.0 # secs

from pygame import mixer
mixer.init() 
beep=mixer.Sound("alarm.wav")
beep.set_volume(0.3)
isSleepy = 0
outputFolder1 = "my_live_output1"

if not os.path.exists(outputFolder1):
    os.makedirs(outputFolder1)
else:
    shutil.rmtree(outputFolder1) 
    os.makedirs(outputFolder1)

outputCounter = 0
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('%s/output_%04d.avi' % (outputFolder1, outputCounter), fourcc, 30.0, (640,480),  isColor=False)
outputCounter11 = outputCounter
text = 'Driver status: nonsleepy.'
do_predict = 0
key = 'a'
def realtime():
    global beep, isSleepy, do_predict, outputCounter11, outputCounter, cap, fourcc, out, key
    hasPredict = False
    #no_count = False
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #if do_predict != 1:
        out.write(gray)
        #else:
            #no_count = True

        if outputCounter % 90 == 0:
            do_predict = 1
        else:
            do_predict = 0
        
        #if no_count == False:
        outputCounter += 1
        #else:
            #no_count = False
        # show the frame
        frame = ps.putBText(frame, text,
        text_offset_x = 20,           
        text_offset_y = 20,              
        vspace = 10,                     
        hspace = 10,
        font_scale = 2,
        font = cv2.FONT_HERSHEY_PLAIN,
        text_RGB = (0, 97, 255), 
        thickness = 2,      
        alpha = 0.2,                     
        background_RGB = (228, 225, 222)
        )
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


    # do a bit of cleanup
    out.release() 
    cv2.destroyAllWindows()
    vs.stop()


def predict():
    global isSleepy, outputCounter11, outputFolder1, clip_start_sec, do_predict, out, outputCounter, key
    global device, model
    while True:
        if do_predict == 1:
            
            print("Predict start...")
            out.release()
            video_path = '%s/output_%04d.avi' % (outputFolder1, outputCounter11)
            print("Predict:",video_path)
            video = EncodedVideo.from_path(video_path)
            outputCounter11 = outputCounter
            out = cv2.VideoWriter('%s/output_%04d.avi' % (outputFolder1, outputCounter), fourcc, 30.0, (640,480),  isColor=False)
            clip_end_sec = clip_start_sec + args.clip_duration
            print("duration is:",clip_start_sec, clip_end_sec)
            video_data = video.get_clip(start_sec = clip_start_sec, end_sec = clip_end_sec)
            video_data = transform(video_data)
            inputs = video_data["video"]
            #inputs = [i.to(device)[None, ...] for i in inputs]
            inputs = torch.Tensor(inputs)
            channel_x, time, h_x, w_x = inputs.shape
            inputs = inputs.reshape(1,channel_x, time, h_x, w_x )
            print(inputs.shape)
            inputs = inputs.to(device)
            with torch.no_grad():
                preds = model(inputs)
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=1).indices
            isSleepy = pred_classes
            print(pred_classes)
            op1 = pred_classes.item()
            path1 = 'C:/Users/User/Desktop/text.txt'
            f = open(path1,'w')
            f.write(str(op1))
            f.close()
            do_predict = 0



def alarm_sound() :
    global beep, text, key
    global isSleepy
    while True:
        if isSleepy == 1:
            text = 'Driver status: sleepy. Wake uppppp!!'
            beep.play()
        else :
            text = 'Driver status: nonsleepy. Good!'
            beep.stop()


"""
class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('my_pub')
        self.publisher_ = self.create_publisher(Twist, 'turtle1/cmd_vel', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        #print("====================+++++++++++++++++===========================ppppp")
        global isSleepy
        vel = Twist()
        t = self.i/10

        if isSleepy == 1 :
            vel.linear.x =  1.0
            vel.angular.z =  1.0
            self.get_logger().info('Publishing: line speed x: %f angular speed z: %f' % (vel.linear.x,vel.angular.z))
        else:
            vel.linear.x =  10.0
            vel.angular.z =  10.0
            
        self.publisher_.publish(vel)
        self.i += 1


def rosNode():
    #print("===============================================ppppp")
    rclpy.init()
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)

"""

a = Thread(target=realtime)
b = Thread(target=alarm_sound)
c = Thread(target=predict)
a.start()
b.start()
c.start()

