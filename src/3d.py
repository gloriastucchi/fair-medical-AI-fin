import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Percorso al file .npz
file_path = '/work3/s232437/fair-medical-AI-fin/data/test/data_2403.npz'
npz_file = np.load(file_path)

# Visualizza le chiavi per controllo
print(list(npz_file.keys()))

# Carica la mappa di spessore RNFL
thickness_map = npz_file['rnflt']  # dovrebbe essere di forma (1, 224, 224)
thickness_map = np.squeeze(thickness_map)  # diventa (224, 224)

# Controlla la forma risultante
print(f"Shape of RNFLT map: {thickness_map.shape}")

# Coordinate X e Y per la griglia
x = np.arange(thickness_map.shape[1])
y = np.arange(thickness_map.shape[0])
X, Y = np.meshgrid(x, y)

# Plot 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, thickness_map, cmap='viridis', edgecolor='none')

ax.set_title('RNFLT 3D Visualization')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Thickness')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.tight_layout()

# Salva l'immagine
output_path = '/work3/s232437/fair-medical-AI-fin/src/rnflt_3d_plot.png'
plt.savefig(output_path)
print(f"Image saved to {output_path}")
