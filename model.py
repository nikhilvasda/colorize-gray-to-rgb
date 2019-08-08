"""
Model architecture of Convolutional Autoencoder for converting grayscale images to RGB images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ConvAutoencoder(nn.Module):
    def __init__(self, batch_size: int, ip_image_dims: int = 32, filter_count: tuple = (16, 32), kernel_dims: int = 3):
        """
        The function requires batch size of the data for initializing the weights of the model.
        :param batch_size: Batch_size of the train/val data.
        :param ip_image_dims: Optional, image dims of the input image.
        :param filter_count: The channel dims of the filters.
        :param kernel_dims: The dims of the filters used to perform convolution operations.
        """
        super(ConvAutoencoder, self).__init__()
        # Parameters.
        self.ip_image_dims = ip_image_dims
        self.batch_size = batch_size
        self.filter_count = filter_count
        self.kernel_dims = kernel_dims
        # Convolution transformations.
        self.conv1 = nn.Conv2d(1, filter_count[0], kernel_dims, padding=1)  # (32-3+2/1)+1=32>>max_pool>>16
        self.conv2 = nn.Conv2d(filter_count[0], filter_count[1], kernel_dims)  # (16-3/1)+1=14>>max_pool>>7
        conv_output_dims = self.get_conv_output_dims(self.ip_image_dims, len(filter_count), 1, pool_dims=2)
        # Linear transformations.
        self.linear1 = nn.Linear(32*conv_output_dims**2, 512)
        self.linear2 = nn.Linear(512, 3072)

    def get_conv_output_dims(self, ip_dims, layer_count, padding, pool_dims):
        """
        Computes the resulting dims of convolution and pool transformations
        :param ip_dims: The dims of the input image
        :param layer_count: Number of conv layers.
        :param padding: If padding is used in the first layer.
        :param pool_dims: The dims of the max_pool transformation.
        :return: Integer value dims of the resulting square image.
        """
        if not layer_count:
            return (ip_dims-self.kernel_dims+2+1)/2
        return self.get_conv_output_dims((ip_dims-self.kernel_dims+2*padding+1)/pool_dims, layer_count-1, 0, pool_dims)

    def forward(self, x):
        """
        The forward function for the the model.
        :param x:
        :return:
        """
        # Encoder Phase
        p1 = functional.max_pool2d(self.conv1(x), (2, 2))
        p2 = functional.max_pool2d(self.conv2(p1), (2, 2))
        # Decoder Phase
        f1 = functional.relu(self.linear1(p2.view(self.batch_size, -1)))
        f2 = torch.sigmoid(self.linear2(f1)).view(-1, 3, self.ip_image_dims, self.ip_image_dims)
        return f2
