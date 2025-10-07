"""
Normalized Dataset Classes for Qlib

This module provides normalized versions of TSDatasetH and TSDataSampler classes
that automatically normalize data before returning it.
"""

from typing import Union, List, Tuple, Optional
from copy import deepcopy
import pandas as pd
import numpy as np
import bisect

from . import TSDatasetH, TSDataSampler
from ...utils import time_to_slc_point


class NormalizedTSDataSampler(TSDataSampler):
    """
    Normalized Time-Series Data Sampler
    
    This class extends TSDataSampler to add data normalization functionality.
    It normalizes the data before returning it in __getitem__ method.
    """
    
    def __init__(self, normalization_method="zscore", **kwargs):
        """
        Initialize the NormalizedTSDataSampler
        
        Parameters
        ----------
        normalization_method : str
            The normalization method to use. Options:
            - "zscore": Standardization (mean=0, std=1)
            - "minmax": Min-Max scaling to [0, 1]
            - "robust": Robust scaling using median and IQR
        **kwargs : dict
            Additional arguments passed to TSDataSampler
        """
        super().__init__(**kwargs)
        self.normalization_method = normalization_method
 
    def _normalize_data(self, data):
        """
        Normalize the input data within each time step (feature-wise normalization)
        
        Parameters
        ----------
        data : np.ndarray
            Input data to normalize
            
        Returns
        -------
        np.ndarray
            Normalized data
        """
        # Check if normalization parameters are available

        
        # Reshape parameters to match data dimensions for broadcasting
        # We want to normalize within each time step (axis=-1 for features)
  
            # Standardization: (x - mean) / std
        data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
        normalized = (data - data_mean) / (data_std + 1e-5)
        normalized = np.clip(normalized, -5, 5)
        return normalized
    
    def __getitem__(self, idx: Union[int, Tuple[object, str], List[int]]):
        """
        Get normalized time-series data
        
        Parameters
        ----------
        idx : Union[int, Tuple[object, str], List[int]]
            Index or indices to retrieve
            
        Returns
        -------
        np.ndarray
            Normalized time-series data
        """
        # Get the original data from parent class
        data = super().__getitem__(idx)
        process_data = data[:, 0:-1]
        # Apply normalization along the time dimension
        normalized_data = self._normalize_data(process_data)
        data[:, 0:-1] = normalized_data
        return data


class NormalizedTSDatasetH(TSDatasetH):
    """
    Normalized Time-Series Dataset Handler
    
    This class extends TSDatasetH to return NormalizedTSDataSampler instances
    with data normalization capabilities.
    """
    
    def __init__(self, normalization_method="zscore", **kwargs):
        """
        Initialize the NormalizedTSDatasetH
        
        Parameters
        ----------
        normalization_method : str
            The normalization method to use
        **kwargs : dict
            Additional arguments passed to TSDatasetH
        """
        self.normalization_method = normalization_method
        super().__init__(**kwargs)
    
    def config(self, **kwargs):
        """
        Configure the dataset
        """
        if "normalization_method" in kwargs:
            self.normalization_method = kwargs.pop("normalization_method")
        super().config(**kwargs)
    
    def _prepare_seg(self, slc: slice, **kwargs) -> NormalizedTSDataSampler:
        """
        Prepare a segment and return a NormalizedTSDataSampler instance
        
        Parameters
        ----------
        slc : slice
            Slice object defining the time segment
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        NormalizedTSDataSampler
            Normalized time-series data sampler
        """
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        if (flt_col := kwargs.pop("flt_col", None)) is None:
            flt_col = self.flt_col

        # TSDatasetH will retrieve more data for complete time-series
        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super(TSDatasetH, self)._prepare_seg(ext_slice, **kwargs)

        flt_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super(TSDatasetH, self)._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        # Create normalized data sampler
        ntsds = NormalizedTSDataSampler(
            data=data,
            start=slc.start,
            end=slc.stop,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
            normalization_method=self.normalization_method
        )
        return ntsds


# Export the classes
__all__ = ["NormalizedTSDatasetH", "NormalizedTSDataSampler"]