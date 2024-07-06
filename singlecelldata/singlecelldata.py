import numpy as np
import pandas as pd
import anndata
from typing import Optional, Union, Dict, List
from scipy import sparse

class SingleCellData:
    def __init__(self, 
                 spectral_data: Optional[np.ndarray] = None, 
                 cell_ids: Optional[np.ndarray] = None, 
                 time_points: Optional[np.ndarray] = None,
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
        omics_data (Optional[Union[np.ndarray, sparse.spmatrix]]): 2D array or sparse matrix with shape (n_cells, n_features).
        omics_var (Optional[pd.DataFrame]): DataFrame with feature information for omics data.
        spatial_coords (Optional[np.ndarray]): 2D array with shape (n_cells, n_dimensions) for spatial coordinates.
        metadata (Optional[Dict]): Dictionary containing additional metadata.
        """
        self.spectral = SingleCellSpectralData(spectral_data, cell_ids, time_points) if spectral_data is not None else None
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

    def add_spectral_layer(self, layer_name: str, layer_data: np.ndarray):
        """
        Add a new layer of spectral data.
        """
        if self.spectral is None:
            raise ValueError("Spectral data not initialized.")
        self.spectral.add_layer(layer_name, layer_data)

    def add_omics_layer(self, layer_name: str, layer_data: Union[np.ndarray, sparse.spmatrix]):
        """
        Add a new layer of omics data.
        """
        if self.omics is None:
            raise ValueError("Omics data not initialized.")
        self.omics.layers[layer_name] = layer_data

    def normalize_spectral(self, layer: Optional[str] = None):
        """
        Normalize spectral data.
        """
        if self.spectral is None:
            raise ValueError("Spectral data not initialized.")
        self.spectral.normalize_data()

    def normalize_omics(self, method: str = 'log1p'):
        """
        Normalize omics data.
        """
        if self.omics is None:
            raise ValueError("Omics data not initialized.")
        if method == 'log1p':
            self.omics.X = np.log1p(self.omics.X)
        # Add other normalization methods as needed

    def dimensionality_reduction(self, method: str = 'pca', n_components: int = 50):
        """
        Perform dimensionality reduction on omics data.
        """
        if self.omics is None:
            raise ValueError("Omics data not initialized.")
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            self.omics.obsm['X_pca'] = pca.fit_transform(self.omics.X)
        # Add other dimensionality reduction methods as needed

    def cluster_cells(self, method: str = 'leiden', resolution: float = 1.0):
        """
        Perform clustering on cells.
        """
        if self.omics is None:
            raise ValueError("Omics data not initialized.")
        if method == 'leiden':
            import scanpy as sc
            sc.pp.neighbors(self.omics)
            sc.tl.leiden(self.omics, resolution=resolution)
        # Add other clustering methods as needed

    def differential_expression(self, groupby: str, method: str = 'wilcoxon'):
        """
        Perform differential expression analysis.
        """
        if self.omics is None:
            raise ValueError("Omics data not initialized.")
        import scanpy as sc
        sc.tl.rank_genes_groups(self.omics, groupby=groupby, method=method)

    def integrate_data(self, method: str = 'harmony'):
        """
        Integrate spectral and omics data.
        """
        if self.spectral is None or self.omics is None:
            raise ValueError("Both spectral and omics data must be initialized.")
        # Implement data integration logic here

    def spatial_analysis(self, method: str = 'moran'):
        """
        Perform spatial analysis.
        """
        if self.spatial_coords is None:
            raise ValueError("Spatial coordinates not available.")
        # Implement spatial analysis logic here

    def visualize(self, plot_type: str, **kwargs):
        """
        Create visualizations.
        """
        if plot_type == 'umap':
            import scanpy as sc
            sc.pl.umap(self.omics, **kwargs)
        elif plot_type == 'spatial':
            # Implement spatial plot
            pass
        elif plot_type == 'spectral':
            # Use SingleCellSpectralData visualization methods
            pass
        # Add other visualization types as needed

    def to_anndata(self) -> anndata.AnnData:
        """
        Convert the SingleCellData to an AnnData object.
        """
        if self.omics is not None:
            adata = self.omics.copy()
        else:
            adata = self.spectral.to_anndata()
        
        if self.spectral is not None:
            adata.layers['spectral'] = self.spectral.spectral_data.reshape(self.spectral.spectral_data.shape[0], -1)
        
        if self.spatial_coords is not None:
            adata.obsm['spatial'] = self.spatial_coords
        
        for key, value in self.metadata.items():
            adata.uns[key] = value
        
        return adata

    @classmethod
    def from_anndata(cls, adata: anndata.AnnData):
        """
        Create a SingleCellData object from an AnnData object.
        """
        spectral_data = adata.layers['spectral'].reshape(-1, adata.uns['n_pixels'], adata.uns['n_channels']) if 'spectral' in adata.layers else None
        omics_data = adata.X
        spatial_coords = adata.obsm['spatial'] if 'spatial' in adata.obsm else None
        
        return cls(spectral_data=spectral_data,
                   cell_ids=adata.obs_names.to_numpy(),
                   time_points=adata.obs['time'].to_numpy() if 'time' in adata.obs else None,
                   omics_data=omics_data,
                   omics_var=adata.var,
                   spatial_coords=spatial_coords,
                   metadata=adata.uns)
