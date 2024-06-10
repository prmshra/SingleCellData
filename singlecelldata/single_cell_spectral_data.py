import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import warnings

class SingleCellSpectralData:
    def __init__(self, spectral_data, cell_ids, n_spectral_channels=None):
        """
        Initialize the SingleCellSpectralData object.
        
        Parameters:
        spectral_data (np.ndarray): 3D array with shape (n_cells, n_pixels, n_spectral_channels).
        cell_ids (np.ndarray): 1D array with shape (n_cells,).
        n_spectral_channels (int): Number of spectral channels. If None, infer from spectral_data.
        """
        self.cell_ids = cell_ids
        n_cells, n_pixels, inferred_n_spectral_channels = spectral_data.shape
        
        if n_spectral_channels is None:
            n_spectral_channels = inferred_n_spectral_channels
        elif n_spectral_channels != inferred_n_spectral_channels:
            raise ValueError("Provided n_spectral_channels does not match the spectral data dimensions.")
        
        # Initialize the AnnData object with spectral data as X
        self.data = ad.AnnData(X=spectral_data.reshape(n_cells, -1))
        
        # Add the pixel and channel dimensions to the AnnData object
        self.data.uns['n_pixels'] = n_pixels
        self.data.uns['n_spectral_channels'] = n_spectral_channels

    def add_layer(self, layer_name, spectral_data_layer):
        """
        Add a new layer to the spectral data.
        
        Parameters:
        layer_name (str): The name of the new layer.
        spectral_data_layer (np.ndarray): 3D array with shape (n_cells, n_pixels, n_spectral_channels).
        """
        n_cells, n_pixels, n_spectral_channels = spectral_data_layer.shape
        
        if (n_cells != self.data.shape[0] or 
            n_pixels != self.data.uns['n_pixels'] or 
            n_spectral_channels != self.data.uns['n_spectral_channels']):
            raise ValueError("Layer dimensions must match the existing spectral data dimensions.")
        
        if not np.array_equal(self.cell_ids, np.arange(n_cells)):
            warnings.warn("Cell IDs do not match. Layer not added.")
            return
        
        self.data.layers[layer_name] = spectral_data_layer.reshape(n_cells, -1)

    def get_image(self, cell_id, channel, layer=None):
        """
        Get the 2D array of pixel intensities for a given cell and spectral channel.
        
        Parameters:
        cell_id (int): The ID of the cell to retrieve the image for.
        channel (int): The spectral channel to retrieve the image for.
        layer (str): The layer name to retrieve data from. Default is None, which uses the main spectral data.
        
        Returns:
        np.ndarray: 2D array of pixel intensities for the given spectral channel.
        """
        idx = np.where(self.cell_ids == cell_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Cell ID {cell_id} not found.")
        idx = idx[0]
        
        n_pixels = self.data.uns['n_pixels']
        if layer:
            spectral_data = self.data.layers[layer][idx]
        else:
            spectral_data = self.data.X[idx]
        
        spectral_data = spectral_data.reshape(n_pixels, self.data.uns['n_spectral_channels'])
        side_length = int(np.sqrt(n_pixels))
        if side_length * side_length != n_pixels:
            raise ValueError("Number of pixels is not a perfect square, cannot reshape to 2D.")
        return spectral_data[:, channel].reshape(side_length, side_length)

    def visualize_image(self, cell_id, channel, layer=None):
        """
        Visualize the 2D array of pixel intensities for a given cell and spectral channel.
        
        Parameters:
        cell_id (int): The ID of the cell to visualize.
        channel (int): The spectral channel to visualize.
        layer (str): The layer name to visualize data from. Default is None, which uses the main spectral data.
        """
        image = self.get_image(cell_id, channel, layer)
        plt.imshow(image, cmap='viridis')
        plt.title(f"Cell ID: {cell_id}, Channel: {channel}, Layer: {layer if layer else 'main'}")
        plt.colorbar(label='Intensity')
        plt.show()
