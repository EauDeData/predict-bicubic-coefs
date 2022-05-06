from ctypes import util
from logging import PlaceHolder
import torch
import torch.nn as nn
import utils
import model
import cv2
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda'
model_instance = model.CoefsModel().to(device)
optimizer = torch.optim.RMSprop(model_instance.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
lossF = utils.image_bicubic_interpolation_loss

EPOCHES = int(5e4)

target = cv2.imread('./example.jpg', cv2.IMREAD_GRAYSCALE) / 255
origin = cv2.resize(target, (64, 64))

X = torch.from_numpy(origin).view(1, origin.shape[0], -1).float().to(device)
Y = torch.from_numpy(target).to(device)
placeholder_coef = torch.from_numpy(np.random.rand(origin.shape[0] * origin.shape[1], 4, 4)).float().to(device)
loss_curve = []
progress = ['·' for i in range(10)]


for i in range(EPOCHES):
    print(f"{''.join(progress)} - EPOCH {i} / {EPOCHES} \t" , end = '\r')
    progress[int((i/EPOCHES * 10)//1)] = 'x'

    optimizer.zero_grad()
    h = model_instance(X)
    loss = lossF(h, Y)

    loss.backward()
    optimizer.step()

    scheduler.step(loss)

    loss_curve.append(loss.item())

print(loss)

fig, axs = plt.subplots(1, 3)
axs[0].plot(loss_curve)
print(origin.shape, placeholder_coef.shape)
img = utils.bicubic_interpolation(torch.from_numpy(origin).to(device), coefs=h)
img =(img - img.min()) / (img.max() - img.min())
axs[1].imshow(img.cpu().detach().numpy() , cmap = 'gray')
axs[1].axis('off')
axs[2].imshow(target, cmap = 'gray')
axs[2].axis('off')

fig.savefig('./state.png')
fig.clf()
