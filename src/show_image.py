from data_handler import EyeFair
import matplotlib.pyplot as plt
import numpy as np
import os

# Create the dataset
dataset = EyeFair(
    data_path='data/train/',
    modality_type='rnflt',
    task='md',
    resolution=224
)

# Get the sample
data_sample, label, attribute = dataset[22]
image = data_sample[0, :, :]

# Print stats
print("Shape:", image.shape)
print("Dtype:", image.dtype)
print("Min:", np.min(image), "Max:", np.max(image), "Mean:", np.mean(image))

# Create the plot
plt.imshow(image)
plt.title("Example Retinal RNFLT Image")
plt.colorbar()

# Save the image
save_path = "/work3/s232437/fair-medical-AI-fin/imageexample/example_image.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.show()
