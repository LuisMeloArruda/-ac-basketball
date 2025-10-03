import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

pd.set_option('display.max_columns', None)  # Show all columns

# Load all files from folder
folder = '../database/'

for file in os.listdir(folder):

    print("\n=================================")
    print(file)

    # Load the dataset
    df = pd.read_csv(folder + file)

    # Get a quick overview
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())

    # Check data types and missing values
    print("\nData types and missing values:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isna().sum())
