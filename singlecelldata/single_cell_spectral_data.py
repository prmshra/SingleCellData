import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict

class SingleCellSpectralData:
    def __init__(self, spectral_data: np.ndarray, cell_ids: np.ndarray, time_points: np.ndarray, n_channels: Optional[int] = None):
        """
        Initialize the SingleCellSpectralData object.

        Parameters:
        spectral_data (np.ndarray): 3D array with shape (n_cells, n_pixels, n_spectral_channels).
        cell_ids (np.ndarray): 1D array with shape (n_cells,).
        time_points (np.ndarray): 1D array with shape (n_cells,).
        n_channels (Optional[int]): Number of spectral channels. If not provided, inferred from spectral_data.
        """
        self.spectral_data = spectral_data  # Store spectral data
        self.cell_ids = cell_ids  # Store cell IDs
        self.time_points = time_points  # Store time points
        
        # If number of spectral channels is not provided, infer from the data
        self.n_channels = n_channels if n_channels is not None else spectral_data.shape[-1]
        
        # Dictionary to store additional layers of spectral data
        self.layers = {}
        
        # Validate initial data
        self._validate_data()

    def _validate_data(self):
        """
        Validate the initial input data for consistency.
        """
        assert len(self.cell_ids) == self.spectral_data.shape[0], "Mismatch between number of cells and spectral data rows."
        assert len(self.time_points) == self.spectral_data.shape[0], "Mismatch between number of cells and time points."
        assert self.spectral_data.shape[-1] == self.n_channels, "Mismatch between spectral data channels and provided n_channels."
        
    def add_layer(self, layer_name: str, layer_data: np.ndarray):
        """
        Add a new layer of spectral data.

        Parameters:
        layer_name (str): Name of the new layer.
        layer_data (np.ndarray): 3D array with shape (n_cells, n_pixels, n_spectral_channels).
        """
        # Ensure new layer has the same shape as existing data, except for the spectral channel dimension
        if layer_data.shape[:-1] != self.spectral_data.shape[:-1]:
            raise ValueError(f"Layer data shape {layer_data.shape} does not match existing data shape {self.spectral_data.shape}.")
        
        # Add the new layer
        self.layers[layer_name] = layer_data

    def show_image(self, cell_id: int, channel: Optional[int] = None, layer: Optional[str] = None, kind: str = 'hyperspectral'):
        """
        Display an image for a specific cell and channel.

        Parameters:
        cell_id (int): ID of the cell to visualize.
        channel (Optional[int]): Spectral channel to visualize. Required if kind is 'hyperspectral'.
        layer (Optional[str]): Name of the layer to visualize. If not provided, defaults to the default layer.
        kind (str): Type of image ('hyperspectral' or 'RGB'). Default is 'hyperspectral'.
        """
        if layer is None and len(self.layers) > 0:
            layer = input("Multiple layers detected. Please specify a layer: ")
        if layer is None or layer == 'default':
            layer = 'default'
            image = self.spectral_data[np.where(self.cell_ids == cell_id)[0][0], :, :]
        else:
            image = self.layers[layer][np.where(self.cell_ids == cell_id)[0][0], :, :]
        
        if kind == 'RGB' and image.shape[-1] >= 3:
            plt.imshow(image[:, :3])
            plt.title(f'Cell ID: {cell_id}, Layer: {layer}, RGB Image')
        else:
            if channel is None:
                channel = int(input(f"Specify the channel to visualize (0 to {self.n_channels - 1}): "))
            plt.imshow(image[:, channel], cmap='viridis')
            plt.title(f'Cell ID: {cell_id}, Channel: {channel}, Layer: {layer}')
        
        plt.colorbar()
        plt.show()

    def get_image(self, cell_id: int, channel: int, layer: str = 'default') -> np.ndarray:
        """
        Get the image data for a specific cell and channel.

        Parameters:
        cell_id (int): ID of the cell.
        channel (int): Spectral channel.
        layer (str): Name of the layer. Defaults to 'default'.

        Returns:
        np.ndarray: 2D array of the image data.
        """
        idx = np.where(self.cell_ids == cell_id)[0][0]
        if layer == 'default':
            return self.spectral_data[idx, :, channel]
        else:
            return self.layers[layer][idx, :, channel]

    def get_hyperspectral_data(self, cell_id: int, layer: Optional[str] = None) -> pd.DataFrame:
        """
        Get the hyperspectral data for a specific cell as a DataFrame.

        Parameters:
        cell_id (int): ID of the cell.
        layer (Optional[str]): Name of the layer. If not provided, defaults to 'default'.

        Returns:
        pd.DataFrame: DataFrame containing the hyperspectral data.
        """
        if layer is None and len(self.layers) > 0:
            layer = input("Multiple layers detected. Please specify a layer: ")
        if layer is None or layer == 'default':
            idx = np.where(self.cell_ids == cell_id)[0][0]
            spectral_data = {f'Channel_{i}': self.spectral_data[idx, :, i] for i in range(self.n_channels)}
        else:
            idx = np.where(self.cell_ids == cell_id)[0][0]
            spectral_data = {f'{layer}_Channel_{i}': self.layers[layer][idx, :, i] for i in range(self.n_channels)}
        
        df = pd.DataFrame(spectral_data)
        return df

    def normalize_data(self):
        """
        Normalize the spectral data across all channels.
        """
        # Normalize spectral data
        self.spectral_data = (self.spectral_data - np.mean(self.spectral_data, axis=0)) / np.std(self.spectral_data, axis=0)
        for layer in self.layers:
            self.layers[layer] = (self.layers[layer] - np.mean(self.layers[layer], axis=0)) / np.std(self.layers[layer], axis=0)

    def to_anndata(self):
        """
        Convert the SingleCellSpectralData to an AnnData object.

        Returns:
        anndata.AnnData: Converted AnnData object.
        """
        import anndata
        obs = pd.DataFrame(index=self.cell_ids)
        obs['time'] = self.time_points
        
        # Use the first spectrum for anndata.X if there's only one layer
        if len(self.layers) == 0:
            X = self.spectral_data.reshape(self.spectral_data.shape[0], -1)
        else:
            first_layer = list(self.layers.keys())[0]
            X = self.layers[first_layer].reshape(self.layers[first_layer].shape[0], -1)
        
        adata = anndata.AnnData(X=X, obs=obs)
        adata.layers['default'] = self.spectral_data.reshape(self.spectral_data.shape[0], -1)
        
        for layer_name, layer_data in self.layers.items():
            adata.layers[layer_name] = layer_data.reshape(layer_data.shape[0], -1)
        
        return adata

    def from_anndata(self, adata: 'anndata.AnnData'):
        """
        Load data from an AnnData object into SingleCellSpectralData.

        Parameters:
        adata (anndata.AnnData): AnnData object to load data from.
        """
        self.cell_ids = adata.obs.index.to_numpy()
        self.time_points = adata.obs['time'].to_numpy()
        self.spectral_data = adata.layers['default'].reshape(len(self.cell_ids), -1, self.n_channels)
        self.layers = {k: v.reshape(len(self.cell_ids), -1, self.n_channels) for k, v in adata.layers.items() if k != 'default'}
