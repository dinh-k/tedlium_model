Adapted from [PyTorch example](https://github.com/pytorch/examples/tree/master/word_language_model).
Modernization and bug fixes applied to [tedlium_model](https://github.com/pwdonh/tedlium_model/tree/newcode2022).

## Problem Statement

The original code was written for PyTorch 0.2.0 and contained several deprecated features that caused:
1. **Infinite recursion errors** in the `repackage_hidden` function
2. **Compatibility issues** with modern PyTorch versions
3. **Deprecation warnings** for removed features

## Root Causes Identified

### 1. Deprecated `Variable` Class
- **Issue**: The code extensively used `torch.autograd.Variable` which was removed in PyTorch 0.4.0
- **Impact**: Caused infinite recursion in `repackage_hidden` function
- **Location**: Multiple files including `func_model.py` and `func_model_data.py`

### 2. Deprecated `volatile` Parameter
- **Issue**: Used `volatile=True/False` parameter which was deprecated and removed
- **Impact**: Caused warnings and potential memory issues
- **Location**: Data processing functions in `func_model_data.py`

### 3. Incorrect Tensor Indexing
- **Issue**: Using `[0]` indexing on 0-dimensional tensors
- **Impact**: Caused `IndexError: invalid index of a 0-dim tensor`
- **Location**: `generate_probabilities.py`

### 4. Implicit Softmax Dimensions
- **Issue**: Missing `dim` parameter in `torch.nn.functional.softmax`
- **Impact**: Deprecation warnings
- **Location**: `generate_probabilities.py`

### 5. Pandas Chained Assignment
- **Issue**: Using chained assignment which will break in pandas 3.0
- **Impact**: FutureWarning messages
- **Location**: `generate_probabilities.py`


Detailed changes documented in the [updated README](https://github.com/dinh-k/tedlium_model/blob/bugfix-modernization/code/README.md).
