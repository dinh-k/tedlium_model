Adapted from [PyTorch example](https://github.com/pytorch/examples/tree/master/word_language_model).
Modernization and bug fixes applied to [tedlium_model](https://github.com/pwdonh/tedlium_model/tree/newcode2022).
This modernization ensures the TED-LIUM model code works reliably with current PyTorch versions while maintaining all original functionality.

## Problem Statement

The original code was written for PyTorch 0.2.0 and contained several deprecated features that caused:
1. **Infinite recursion errors** in the `repackage_hidden` function
2. **Compatibility issues** with modern PyTorch versions
3. **Deprecation warnings** for removed features

## Key Benefits

1. **Eliminated Recursion Errors**: Fixed infinite recursion in `repackage_hidden`
2. **Modern PyTorch Support**: Removed all deprecated features
3. **Improved Performance**: Better memory management with `torch.no_grad()`
4. **Future-Proof**: Compatible with upcoming pandas 3.0
5. **Cleaner Output**: Reduced warning messages
6. **Maintained Functionality**: All original features preserved

## Files Modified

1. `func_model.py` - Core model updates and `repackage_hidden` fix
2. `func_model_data.py` - Data processing modernization
3. `generate_probabilities.py` - Main script updates and error fixes
4. Created `../output/` directory for file saving

## Compatibility

- **PyTorch**: Compatible with versions 1.0+ (tested with current versions)
- **Python**: 3.6+ (recommended 3.8+)
- **Pandas**: 1.0+ (with proper indexing)
- **NumPy**: 1.16+

All detailed changes are documented in [README](https://github.com/dinh-k/tedlium_model/blob/bugfix-modernization/code/README.md).
