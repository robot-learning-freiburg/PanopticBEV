print('Checking torch')
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())

print('Checking opencv')
import cv2

print('Checking Pillow')
import PIL

print('Checking kornia')
import kornia