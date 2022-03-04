import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

#NOTE: got this for some forum, used to find that child 7 is the last conv layer
#child_counter = 0
#for child in model.children():
#   print(" child", child_counter, "is:")
#  print(child)
#   child_counter += 1
"""
#4c did not manage to make it work
i = 0
for child in model.children():
    if i == 6: #conv layer at layer 7 / index 6
        last_clayer = child
    i = i + 1

act1 = last_clayer.conv1(image) #1
act2 = last_clayer.conv2(image) #2
act3 = last_clayer.conv3(image) #3
act4 = last_clayer.conv4(image) #4
act5 = last_clayer.conv5(image) #5
act6 = last_clayer.conv6(image) #6
act7 = last_clayer.conv7(image) #7
act8 = last_clayer.conv8(image) #8
act9 = last_clayer.conv9(image) #9
act10 = last_clayer.conv10(image) #10

acts = [act1, act2, act3, act4, act5, act6, act7, act8, act9, act10]

for i in range(10):
    act_image = torch_image_to_numpy(acts[i])
    plt.subplot(1, 10, i+1)
    plt.imshow(act_image)

plt.savefig("4c.png")
"""

indices = [14, 26, 32, 49, 52]
num_f = len(indices)

for i in range(num_f):
    plt.subplot(2, num_f, i + 1)
    F = torch_image_to_numpy(first_conv_layer.weight[indices[i], :, :, :])
    plt.imshow(F)
    plt.title(f"{indices[i]}")
    if i == 0:
        plt.ylabel('Filter')

    plt.subplot(2, num_f, i + 6)
    A = torch_image_to_numpy(activation[0, indices[i], :, :])
    plt.imshow(A)
    if i == 0:
        plt.ylabel('Activations')

#plt.savefig("task4b.png")
#plt.show()