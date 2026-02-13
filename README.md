# NumPy Learning Repository

Welcome to the NumPy tutorial notebook! This repository contains comprehensive examples and explanations for learning NumPy, a fundamental package for scientific computing with Python.

## Overview

NumPy (Numerical Python) is a powerful library that provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. This tutorial covers everything from basic array creation to advanced topics like broadcasting and vectorization.

## Topics Covered

### 1. Array Creation
- Creating arrays using `np.array()`
- Using `np.arange()`, `np.linspace()`, `np.logspace()`
- Creating special arrays: `np.zeros()`, `np.ones()`, `np.full()`, `np.empty()`

### 2. Random Number Generation
- `np.random.rand()` - Random floats between 0 and 1
- `np.random.randn()` - Random numbers from standard normal distribution
- `np.random.randint()` - Random integers
- `np.random.normal()` - Normal distribution with custom parameters

### 3. Data Types & Type Casting
- Understanding NumPy data types
- Converting between types using `.astype()`

### 4. Multidimensional Arrays
- Creating 2D and multi-dimensional arrays
- Understanding `ndim`, `shape`, `size`, and `itemsize`

### 5. Array Reshaping
- `reshape()` - Change array shape
- `ravel()` - Flatten array (view)
- `flatten()` - Flatten array (copy)

### 6. Array Slicing & Indexing
- Basic slicing
- Advanced indexing with lists
- Using `np.take()`

### 7. Iterating Arrays
- `np.nditer()` - Iterate over elements
- `np.ndenumerate()` - Get indices and values

### 8. View vs Copy
- Understanding the difference between view and copy
- How modifications affect original arrays

### 9. Array Operations
- Transpose and swap axes
- Arithmetic operations on arrays

### 10. Joining & Splitting
- `concatenate()`, `vstack()`, `hstack()`, `stack()`
- `split()`, `vsplit()`, `hsplit()`

### 11. Repetition
- `np.repeat()` - Repeat each element
- `np.tile()` - Repeat entire array

### 12. Aggregate Functions
- Statistical functions: `median()`, `mean()`, `min()`, `max()`
- `sum()`, `average()`, `std()`, `var()`
- Cumulative operations: `cumsum()`, `cumprod()`

### 13. Conditional Operations
- `np.where()` - Conditional selection
- Logical operations: `logical_and()`, `logical_or()`, `logical_not()`, `logical_xor()`

### 14. Broadcasting
- Understanding NumPy broadcasting rules
- Performing operations on arrays with different shapes

### 15. Vectorization
- Performance benefits over traditional loops
- Memory-efficient operations

### 16. Memory Comparison
- Comparing memory usage between Python lists and NumPy arrays

### 17. Function Vectorization
- Applying custom functions to array elements using `np.vectorize()`

### 18. Handling Missing Values
- Working with `np.nan`, `np.inf`
- Using `np.nan_to_num()` to handle missing values

## Requirements

- Python 3.x
- NumPy library

## Installation

```
bash
pip install numpy
```

## Usage

You can run the Jupyter notebook `libraries/Numpy.ipynb` to explore all the examples interactively. Each code cell demonstrates a specific NumPy concept with detailed explanations.

## Why NumPy?

- **Performance**: NumPy arrays are more efficient than Python lists for numerical operations
- **Broadcasting**: Perform operations on arrays with different shapes
- **Vectorization**: Avoid explicit loops for better performance
- **Memory Efficiency**: NumPy arrays use less memory than Python lists

## Learning Path

This notebook is designed for:
1. Beginners learning NumPy for the first time
2. Data Science enthusiasts
3. Anyone working with numerical data in Python

## Contributing

Feel free to explore, modify, and learn from this notebook. Happy learning!

## License

This project is open source and available for educational purposes.
