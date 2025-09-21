# TED-LIUM Model Code - Modernization and Bug Fixes

## Overview

This document details the changes made to modernize the TED-LIUM model code from PyTorch 0.2.0 to current PyTorch versions and fix critical recursion errors.

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

## Detailed Changes Made

### 1. `func_model.py` - Core Model Updates

#### A. Updated `repackage_hidden` Function
**Before:**
```python
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
```

**After:**
```python
def repackage_hidden(h):
    """Wraps hidden states in new tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
```

**Changes:**
- Replaced `Variable` usage with modern tensor operations
- Changed `type(h) == Variable` to `isinstance(h, torch.Tensor)`
- Replaced `Variable(h.data)` with `h.detach()`
- Updated docstring to reflect tensor usage

#### B. Updated `init_hidden` Methods
**Before:**
```python
def init_hidden(self, bsz):
    weight = next(self.parameters()).data
    if self.rnn_type == 'LSTM':
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
    else:
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
```

**After:**
```python
def init_hidden(self, bsz):
    weight = next(self.parameters()).data
    if self.rnn_type == 'LSTM':
        return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_())
    else:
        return weight.new(self.nlayers, bsz, self.nhid).zero_()
```

**Changes:**
- Removed `Variable` wrapper around tensor creation
- Maintained proper device placement using `weight.new()`
- Applied to all model classes: `RNNModel`, `TedliumModel`, `TedliumModelCombined`, `TedliumModelCombined2`

#### C. Removed Variable Import
**Before:**
```python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
```

**After:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### 2. `func_model_data.py` - Data Processing Updates

#### A. Updated `data_producer_combined` Function
**Before:**
```python
for (ii, pointer) in enumerate(pointers):
    x = [Variable(d[pointer:pointer+seq_length,:], volatile=evaluation) for d in datanp]
    x_d = Variable(datanp_d[pointer:pointer+seq_length,:,:], volatile=evaluation)
    y = [Variable(d[pointer+1:pointer+seq_length+1,:], volatile=evaluation) for d in datanp]
    yield ((x[3],x[0],x_d),(y[3].view(-1),y[1].view(-1),y[2].view(-1)),ii)
```

**After:**
```python
for (ii, pointer) in enumerate(pointers):
    x = [d[pointer:pointer+seq_length,:] for d in datanp]
    x_d = datanp_d[pointer:pointer+seq_length,:,:]
    y = [d[pointer+1:pointer+seq_length+1,:] for d in datanp]
    yield ((x[3],x[0],x_d),(y[3].view(-1),y[1].view(-1),y[2].view(-1)),ii)
```

**Changes:**
- Removed `Variable` wrapper and `volatile` parameter
- Simplified tensor operations
- Maintained same data structure for compatibility

#### B. Updated `data_producer` Function
**Before:**
```python
yield ((Variable(x, volatile=evaluation), Variable(x_d, volatile=evaluation)),
       (Variable(y.view(-1), volatile=evaluation),
        Variable(y_d.view(-1), volatile=evaluation)), ii)
```

**After:**
```python
yield ((x, x_d),
       (y.view(-1),
        y_d.view(-1)), ii)
```

#### C. Removed Variable Import
**Before:**
```python
import torch
import numpy as np
from torch.autograd import Variable
```

**After:**
```python
import torch
import numpy as np
```

### 3. `generate_probabilities.py` - Main Script Updates

#### A. Added `torch.no_grad()` Context Manager
**Before:**
```python
output, hidden = model(data, hidden)
loss, loss_phone, loss_word = model.criterion(output, targets)
total_loss += loss.data
phone_loss += loss_phone.data
word_loss += loss_word.data
hidden = repackage_hidden(hidden)
```

**After:**
```python
with torch.no_grad():
    output, hidden = model(data, hidden)
    loss, loss_phone, loss_word = model.criterion(output, targets)
    total_loss += loss.data
    phone_loss += loss_phone.data
    word_loss += loss_word.data
    hidden = repackage_hidden(hidden)
```

**Changes:**
- Added `torch.no_grad()` context manager to replace `volatile` functionality
- Ensures no gradients are computed during evaluation
- Improves memory efficiency

#### B. Fixed Tensor Indexing
**Before:**
```python
results = (total_loss[0]/(batch+1), phone_loss[0]/(batch+1), word_loss[0]/(batch+1))
```

**After:**
```python
results = (total_loss.item()/(batch+1), phone_loss.item()/(batch+1), word_loss.item()/(batch+1))
```

**Changes:**
- Replaced `[0]` indexing with `.item()` for 0-dimensional tensors
- Fixed `IndexError: invalid index of a 0-dim tensor`

#### C. Added Softmax Dimension Parameter
**Before:**
```python
p_phones = torch.nn.functional.softmax(output[0].view(-1,ntokens_phone)).data.numpy()
p_words[indices,:] = torch.nn.functional.softmax(output[1].view(-1,ntokens_word)).data.numpy()
```

**After:**
```python
p_phones = torch.nn.functional.softmax(output[0].view(-1,ntokens_phone), dim=1).data.numpy()
p_words[indices,:] = torch.nn.functional.softmax(output[1].view(-1,ntokens_word), dim=1).data.numpy()
```

**Changes:**
- Added `dim=1` parameter to `softmax` calls
- Fixed deprecation warnings about implicit dimension choice

#### D. Fixed Pandas Chained Assignment
**Before:**
```python
df.iloc[-1]['target_phone'] = ''
df.iloc[-1]['target_word'] = ''
df.iloc[-1]['surprise'] = np.nan
```

**After:**
```python
df.loc[df.index[-1], 'target_phone'] = ''
df.loc[df.index[-1], 'target_word'] = ''
df.loc[df.index[-1], 'surprise'] = np.nan
```

**Changes:**
- Replaced chained assignment with proper `.loc` indexing
- Fixed FutureWarning about pandas 3.0 compatibility

### 4. Environment and Setup Changes

#### A. Created Output Directory
```bash
mkdir ..\output
```
- Created missing output directory to prevent file saving errors

## Testing and Validation

### Before Fixes
- Infinite recursion error in `repackage_hidden`
- `IndexError: invalid index of a 0-dim tensor`
- Multiple deprecation warnings
- `OSError: Cannot save file into a non-existent directory`

### After Fixes
- Script runs successfully without recursion errors
- All 11 output files generated (5MB-24MB each)
- Modern PyTorch compatibility
- Cleaner warning output
- Proper file saving

## Output Files Generated

The script successfully processed all 11 TED talks and generated:
- `EricMead_2009P_phone_outputs.csv` (7.2MB)
- `MichaelSpecter_2010_phone_outputs.csv` (15MB)
- `JaneMcGonigal_2010_phone_outputs.csv` (20MB)
- `TomWujec_2010U_phone_outputs.csv` (6.3MB)
- `GaryFlake_2010_phone_outputs.csv` (5.6MB)
- `RobertGupta_2010U_phone_outputs.csv` (5.1MB)
- `DanBarber_2010_phone_outputs.csv` (13MB)
- `JamesCameron_2010_phone_outputs.csv` (15MB)
- `BillGates_2010_phone_outputs.csv` (24MB)
- `DanielKahneman_2010_phone_outputs.csv` (17MB)
- `AimeeMullins_2009P_phone_outputs.csv` (16MB)

## Usage

To run the modernized script:

```bash
cd tedlium_model/code
python generate_probabilities.py
```

## Compatibility

- **PyTorch**: Compatible with versions 1.0+ (tested with current versions)
- **Python**: 3.6+ (recommended 3.8+)
- **Pandas**: 1.0+ (with proper indexing)
- **NumPy**: 1.16+ (for data processing)

## Key Benefits

1. **Eliminated Recursion Errors**: Fixed infinite recursion in `repackage_hidden`
2. **Modern PyTorch Support**: Removed all deprecated features
3. **Improved Performance**: Better memory management with `torch.no_grad()`
4. **Future-Proof**: Compatible with upcoming pandas 3.0
5. **Cleaner Output**: Reduced warning messages
6. **Maintained Functionality**: All original features preserved

## Technical Notes

- The `repackage_hidden` function now properly handles both single tensors and tuples of tensors
- Hidden state initialization uses proper device placement
- Data processing maintains the same interface for compatibility
- All model classes (RNNModel, TedliumModel, etc.) have been updated consistently

## Files Modified

1. `func_model.py` - Core model updates and `repackage_hidden` fix
2. `func_model_data.py` - Data processing modernization
3. `generate_probabilities.py` - Main script updates and error fixes
4. Created `../output/` directory for file saving

This modernization ensures the TED-LIUM model code works reliably with current PyTorch versions while maintaining all original functionality.

