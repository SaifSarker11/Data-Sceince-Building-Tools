# Data Science Building Tools

A comprehensive educational resource for learning essential Python libraries used in data science.

## 📚 Overview

This repository contains interactive Jupyter notebooks covering fundamental data science tools including **NumPy**, **Pandas**, and **Matplotlib**. Each notebook provides hands-on examples, code snippets, and explanations for working with data in Python.

## 🗂️ Project Structure

```
Data-Sceince-Building-Tools/
├── README.md
└── libraries/
    ├── Numpy.ipynb        # NumPy library tutorials
    ├── Pandas.ipynb       # Pandas library tutorials
    └── Matplotlib.ipynb   # Matplotlib library tutorials
```

## 📖 Contents

### NumPy (`libraries/Numpy.ipynb`)
Learn the fundamentals of numerical computing with NumPy:

- **Array Creation**: `np.array()`, `np.arange()`, `np.linspace()`, `np.logspace()`, `np.zeros()`, `np.ones()`, `np.full()`, `np.empty()`
- **Random Number Generation**: `np.random.rand()`, `np.random.randn()`, `np.random.randint()`, `np.random.normal()`
- **Data Types & Type Casting**: Understanding NumPy data types and converting between them
- **Multidimensional Arrays**: Creating and working with n-dimensional arrays
- **Reshaping**: `reshape()`, `ravel()`, `flatten()`
- **Array Slicing**: Indexing and slicing techniques for 1D and multidimensional arrays
- **Iteration**: Using `np.nditer()` and `np.ndenumerate()`
- **View vs Copy**: Understanding the difference between views and copies
- **Matrix Operations**: Transpose, swapaxes
- **Arithmetic Operations**: Addition, subtraction, multiplication, division, exponent
- **Trigonometric Functions**: `np.sin()`, `np.exp()`, etc.
- **Joining & Splitting**: `concatenate()`, `vstack()`, `hstack()`, `split()`, `vsplit()`, `hsplit()`
- **Aggregate Functions**: `sum()`, `mean()`, `median()`, `min()`, `max()`, `std()`, `var()`, `cumsum()`, `cumprod()`
- **Conditional Operations**: `np.where()`, `np.logical_and()`, `np.logical_or()`, `np.logical_not()`
- **Broadcasting**: Understanding how NumPy performs operations on arrays of different shapes
- **Vectorization**: Performance optimization using vectorized operations
- **Memory Comparison**: List vs NumPy array memory usage
- **Handling Missing Values**: Working with `nan`, `inf`, and `-inf`

### Pandas (`libraries/Pandas.ipynb`)
Master data manipulation and analysis with Pandas:

- **Pandas Series**:
  - Attributes: `dtype`, `name`, `index`, `values`, `shape`, `size`
  - Methods: `sort_values()`, `sort_index()`, `value_counts()`, `sum()`, `max()`, `min()`, `mean()`, `median()`, `std()`, `var()`
- **Indexing Techniques**:
  - Position-based: `iloc[]`
  - Label-based: `loc[]`
  - Conditional indexing
- **Series from Dictionaries**: Creating series from key-value pairs
- **DataFrames**:
  - Creation and manipulation
  - Selection: `head()`, `tail()`, `iloc`, `loc`
  - Shape, info, and describe
- **Data Cleaning**:
  - Handling missing values: `dropna()`, `fillna()`
  - Removing duplicates: `drop_duplicates()`
  - Replacing values: `replace()`
- **Column Operations**:
  - Renaming columns
  - Adding new columns
  - Splitting columns
- **Lambda Functions**: Applying custom functions to data
- **Joins & Merges**:
  - Concatenation: `pd.concat()`
  - Merging: `pd.merge()` with inner, left, right, and outer joins
- **Importing Files**: Reading Excel files with `pd.read_excel()`

### Matplotlib (`libraries/Matplotlib.ipynb`)
Create stunning data visualizations with Matplotlib:

- **Univariate (Numerical)**:
  - **Line Plot**: `plt.plot()` with markers, colors, and linestyles
  - **Histogram**: `plt.hist()` for frequency distribution
  - **Box Plot**: `plt.boxplot()` for statistical summaries
- **Univariate (Categorical)**:
  - **Bar Chart**: `plt.bar()` and `plt.barh()` for categorical comparisons
  - **Pie Chart**: `plt.pie()` for proportional data
- **Bivariate**:
  - **Scatter Plot**: `plt.scatter()` for relationship between two variables
  - **Multi-group Box Plot**: Comparing distributions across categories
- **Plot Customization**:
  - Titles, axis labels, grids
  - Colors, markers, linestyles
  - Legends and annotations
- **Subplots**: `plt.subplot()` and `plt.subplots()` for multi-panel figures
- **Object-Oriented API**: Using `Figure` and `Axes` objects for fine-grained control
- **Interactive 3D Plots**: `px.scatter_3d()` using Plotly Express

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab
- Required packages:

```bash
pip install numpy pandas matplotlib plotly nbformat
```

### Running the Notebooks
1. Clone or download this repository
2. Navigate to the project directory
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the desired notebook from the `libraries/` folder
5. Run the cells to learn and experiment

## 💡 Learning Tips

- Start with **NumPy** to understand the basics of numerical computing in Python
- Move on to **Pandas** to learn data manipulation and analysis
- Finish with **Matplotlib** to visualize your data effectively
- Run each cell in the notebooks to see the output
- Modify the code examples to experiment with different parameters
- Refer to the official documentation for more in-depth information

## 📚 Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Plotly Documentation](https://plotly.com/python/)
- [Python Official Documentation](https://docs.python.org/3/)

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

---

Happy Learning! 📊🐼📈
