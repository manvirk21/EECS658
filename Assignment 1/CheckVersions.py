'''
Name: EECS 658 Assignment 1
Author: Manvir Kaur
KUID: 3064194
Date: 08/31/2023
Purpose: Check versions of Python & create ML “Hello World!” program.
Output: Prints out the versions of Python, scipy, numpy, pandas, and sklearn
        Prints out “Hello World!”
Sources: Dr. Johnson Assignment 1 instruction shet
'''

# Import necessary libraries
import sys
import scipy
import numpy
import pandas
import sklearn

# Print Python version
print('Python: {}'.format(sys.version))

# Print scipy version
print('scipy: {}'.format(scipy.__version__))

# Print numpy version
print('numpy: {}'.format(numpy.__version__))

# Print pandas version
print('pandas: {}'.format(pandas.__version__))

# Print scikit-learn version
print('sklearn: {}'.format(sklearn.__version__))

# Print "Hello World!"
print("\nHello World!")
