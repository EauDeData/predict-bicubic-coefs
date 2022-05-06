import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2

device = 'cuda'
X_MIDDLE_POINTS = torch.tensor([1, 0.5, 0.5**2, 0.5**3]).to(device)
Y_MIDDLE_POINTS = torch.tensor([[1, 0.5, 0.5**2, 0.5**3]]).view(-1).to(device)

def bicubic_interpolation(origin, coefs = None):
    '''
    expected coefs shape = [bs, origin.nPixels, 4, 4];
    
    '''
    if type(coefs) == type(None): coefs = torch.zeros(origin.shape[0]*origin.shape[1], 4, 4)
    xCoefs = torch.matmul(X_MIDDLE_POINTS, coefs)
    pixels = torch.matmul(xCoefs, Y_MIDDLE_POINTS)

    target = torch.zeros(origin.shape[0] * 2, origin.shape[1] * 2).to(device)
    target[::2, ::2] += origin
    target[1::2, 1::2] += pixels.reshape(origin.shape[0], origin.shape[1])
    return target

def image_bicubic_interpolation_loss(coefs, target):
    
    xCoefs = torch.matmul(X_MIDDLE_POINTS, coefs)
    pixels = torch.matmul(xCoefs, Y_MIDDLE_POINTS)

    residual = target[1::2, 1::2].reshape(-1) - pixels
    squared = residual ** 2
    return torch.sum(squared, dim = 0) / pixels.shape[-1]



if __name__ == '__main__':
    input_img = cv2.imread('./bricks.jpg', cv2.IMREAD_GRAYSCALE)
    print(input_img.shape)
    transfrom = bicubic_interpolation(torch.from_numpy(input_img).to(device), coefs=torch.from_numpy(np.random.rand(input_img.shape[0] * input_img.shape[1], 4, 4)).float().to(device))
    plt.imshow(transfrom.numpy(), cmap = 'gray')
    plt.show()
    
    #coefs = torch.random(input_img.shape[0]*input_img.shape[1], 4, 4)
    #target = cv2.resize(input_img, (input_img.shape[0], input_img.shape[1])) # torch.ones(input_img.shape[0] * 2, input_img.shape[1] * 2)
    #print(image_bicubic_interpolation_loss(coefs, target))
