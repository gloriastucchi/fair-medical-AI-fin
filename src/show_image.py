from data_handler import EyeFair
import matplotlib.pyplot as plt

# Create the dataset
dataset = EyeFair(
    data_path='/Users/gloriastucchi/Desktop/dataset/train',
    modality_type='rnflt',  # ensures we load the 'rnflt' key from each .npz
    task='md',
    resolution=224
)

# Get the first sample (returns: data_sample, y, attr)
data_sample, label, attribute = dataset[22]
# data_sample might have shape (1, 224, 224) or (depth, 224, 224)
image = data_sample[0, :, :]  # if there's a single channel, pick channel 0
import matplotlib.pyplot as plt
import numpy as np

# Print shape, dtype, min, max, mean
print("Shape:", image.shape)
print("Dtype:", image.dtype)
print("Min:", np.min(image), "Max:", np.max(image), "Mean:", np.mean(image))

# Display the image
plt.imshow(image)
plt.title("Example Retinal RNFLT Image")
plt.colorbar()  # Optional: shows the value scale
plt.show()
