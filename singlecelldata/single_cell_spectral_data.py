import numpy as np
import matplotlib.pyplot as plt

class SingleCellSpectralData:
    def __init__(self, spectral_data, cell_ids, times=None):
        """
        Initialize the SingleCellSpectralData object.
        
        Parameters:
        spectral_data (np.ndarray): 4D array with shape (n_cells, width, height, n_spectral_channels).
        cell_ids (np.ndarray): 1D array with shape (n_cells,).
        times (np.ndarray): 1D array with shape (n_cells,), optional time stamps for each cell.
        """
        self.spectral_data = spectral_data
        self.cell_ids = cell_ids
        self.times = times if times is not None else np.zeros(len(cell_ids))

    def get_image(self, cell_id, channel):
        """
        Get the 2D array of pixel intensities for a given cell and spectral channel.
        
        Parameters:
        cell_id (int): The ID of the cell to retrieve the image for.
        channel (int): The spectral channel to retrieve the image for.
        
        Returns:
        np.ndarray: 2D array of pixel intensities for the given spectral channel.
        """
        idx = np.where(self.cell_ids == cell_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Cell ID {cell_id} not found.")
        idx = idx[0]
        return self.spectral_data[idx, :, :, channel]

    def visualize_image(self, cell_id, channel):
        """
        Visualize the 2D array of pixel intensities for a given cell and spectral channel.
        
        Parameters:
        cell_id (int): The ID of the cell to visualize.
        channel (int): The spectral channel to visualize.
        """
        image = self.get_image(cell_id, channel)
        plt.imshow(image, cmap='viridis')
        plt.title(f"Cell ID: {cell_id}, Channel: {channel}")
        plt.colorbar(label='Intensity')
        plt.show()
