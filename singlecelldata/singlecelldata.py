import numpy as np
import pandas as pd
import anndata
from typing import Optional, Union, Dict, List
from scipy import sparse
import xarray as xr
from datetime import datetime

# Import SingleCellSpectralData from the separate file
from .singlecellspectraldata import SingleCellSpectralData

class SingleCellData:
    def __init__(self, 
                 spectral_data: Optional[np.ndarray] = None, 
                 cell_ids: Optional[np.ndarray] = None, 
                 time_points: Optional[np.ndarray] = None,
                 n_channels: Optional[int] = None,
                 omics_data: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
                 omics_var: Optional[pd.DataFrame] = None,
                 spatial_coords: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize the SingleCellData object.

        Parameters:
        spectral_data (Optional[np.ndarray]): 3D array with shape (n_cells, n_pixels, n_spectral_channels).
        cell_ids (Optional[np.ndarray]): 1D array with shape (n_cells,).
        time_points (Optional[np.ndarray]): 1D array with shape (n_cells,).
        n_channels (Optional[int]): Number of spectral channels.
        omics_data (Optional[Union[np.ndarray, sparse.spmatrix]]): 2D array or sparse matrix with shape (n_cells, n_features).
        omics_var (Optional[pd.DataFrame]): DataFrame with feature information for omics data.
        spatial_coords (Optional[np.ndarray]): 2D array with shape (n_cells, n_dimensions) for spatial coordinates.
        metadata (Optional[Dict]): Dictionary containing additional metadata.
        """
        self.spectral = SingleCellSpectralData(spectral_data, cell_ids, time_points, n_channels) if spectral_data is not None else None
        self.omics = anndata.AnnData(X=omics_data, var=omics_var) if omics_data is not None else None
        self.spatial_coords = spatial_coords
        self.metadata = metadata or {}
        
        self._validate_data()

    def _validate_data(self):
        """
        Validate the consistency of data across different modalities.
        """
        n_cells = set()
        if self.spectral is not None:
            n_cells.add(len(self.spectral.cell_ids))
        if self.omics is not None:
            n_cells.add(self.omics.n_obs)
        if self.spatial_coords is not None:
            n_cells.add(len(self.spatial_coords))
        
        if len(n_cells) > 1:
            raise ValueError("Mismatch in number of cells across data modalities.")

    # Methods that interact with SingleCellSpectralData
    def add_spectral_layer(self, layer_name: str, layer_data: np.ndarray):
        if self.spectral is None:
            raise ValueError("Spectral data not initialized.")
        self.spectral.add_layer(layer_name, layer_data)

    def show_spectral_image(self, cell_id: int, channel: Optional[int] = None, layer: Optional[str] = None, kind: str = 'hyperspectral'):
        if self.spectral is None:
            raise ValueError("Spectral data not initialized.")
        self.spectral.show_image(cell_id, channel, layer, kind)

    def get_spectral_image(self, cell_id: int, channel: int, layer: str = 'default') -> np.ndarray:
        if self.spectral is None:
            raise ValueError("Spectral data not initialized.")
        return self.spectral.get_image(cell_id, channel, layer)

    def get_hyperspectral_data(self, cell_id: int, layer: Optional[str] = None) -> pd.DataFrame:
        if self.spectral is None:
            raise ValueError("Spectral data not initialized.")
        return self.spectral.get_hyperspectral_data(cell_id, layer)

    def normalize_spectral_data(self):
        if self.spectral is None:
            raise ValueError("Spectral data not initialized.")
        self.spectral.normalize_data()

    # Methods for omics data
    def add_omics_layer(self, layer_name: str, layer_data: Union[np.ndarray, sparse.spmatrix]):
        if self.omics is None:
            raise ValueError("Omics data not initialized.")
        self.omics.layers[layer_name] = layer_data

    # Methods for integrated analysis (to be implemented)
    def integrate_data(self):
        # Implement data integration logic here
        pass

    def dimensionality_reduction(self, method: str = 'pca', n_components: int = 50):
        # Implement dimensionality reduction for integrated data
        pass

    def cluster_cells(self, method: str = 'leiden', resolution: float = 1.0):
        # Implement clustering for integrated data
        pass

    # Conversion methods
    def to_anndata(self) -> anndata.AnnData:
        if self.spectral is not None:
            adata = self.spectral.to_anndata()
        elif self.omics is not None:
            adata = self.omics.copy()
        else:
            raise ValueError("No data available to convert to AnnData.")

        if self.spatial_coords is not None:
            adata.obsm['spatial'] = self.spatial_coords

        for key, value in self.metadata.items():
            adata.uns[key] = value

        return adata

    @classmethod
    def from_anndata(cls, adata: anndata.AnnData):
        spectral_data = adata.layers['default'].reshape(-1, adata.uns['n_pixels'], adata.uns['n_channels']) if 'default' in adata.layers else None
        omics_data = adata.X
        spatial_coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else None

        return cls(spectral_data=spectral_data,
                   cell_ids=adata.obs_names.to_numpy(),
                   time_points=adata.obs['time'].to_numpy() if 'time' in adata.obs else None,
                   n_channels=adata.uns['n_channels'] if 'n_channels' in adata.uns else None,
                   omics_data=omics_data,
                   omics_var=adata.var,
                   spatial_coords=spatial_coords,
                   metadata=adata.uns)

    # Additional methods can be added here for saving, loading, visualization, etc.
