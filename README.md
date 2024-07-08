# Enhancing Shopping Experience through Data-Driven Product Recommendations

This repository aims to develop a robust recommendation system for H&M Group by analyzing transaction data, customer information, and article details. H&M Group operates globally with 53 online markets and around 4850 stores. By leveraging data science techniques, we strive to uncover insights for effective product recommendations, enhance the shopping experience, reduce returns, and support sustainability. Our goal is to improve customer satisfaction and engagement through personalized recommendations.

## Executive Summary

### Problem
Customers often face difficulty finding suitable fashion items that match their preferences and needs, leading to a suboptimal shopping experience.

### Solution
To address this issue, we have developed a personalized recommendation system. By analyzing customer preferences and purchasing patterns, the system can provide more accurate and tailored product recommendations.

### Impact
The implementation of this personalized recommendation system is expected to significantly enhance the shopping experience, leading to increased customer satisfaction and engagement. Additionally, by providing better-suited recommendations, the system can help reduce the number of returns, contributing to more sustainable shopping practices.

## Process Overview

- **Data processing steps**: Cleaning and preprocessing the data, handling null values, and dropping redundant columns.
- **Modeling approaches**: Evaluating various modeling and predictive methods to build the recommendation system.
- **Prototyping directions**: Developing prototypes to test and refine the recommendation system.

## Models Implemented

1. **User-Item Matrix Recommender System**
2. **Matrix Factorization Recommender System**
3. **BERT Recommender System**

## Organization

### Repository Structure

- **environment_data_links/**: Stores the `hm_env.yml` and `data-links.md` files for setting up the environment and accessing data.
- **notebooks/**: Includes all final Jupyter notebooks used in the project.
- **presentations/**: Contains presentations summarizing the project.
- **.gitignore**: Lists files and folders to be ignored by Git version control.
- **README.md**: Project landing page (this file).

## Environment Setup

To replicate the project environment, follow these steps:

1. **Download the Conda environment file**: You can download the environment YAML file from the `environment_data_links` folder, it is stored as `hm_env.yml`.
2. **Install the Conda environment**: Ensure you have Conda installed. Then, run:
    ```bash
    conda env create -f path_to_downloaded_hm_env.yml
    conda activate hm_env
    ```

## Dataset

- **Links to Datasets**: The datasets required for this project are stored in publicly accessible cloud storage. Please refer to the `data-links.md` file in the `environment_data_links/` folder for the links and instructions to download the datasets.

Note: Due to the large size of the images, they have not yet been uploaded to this repository.
