# Enhancing Clothing Recommendations through Data-Driven Recommender Systems

This repository documents the development of robust recommender systems by analyzing H&M Group transactional data, customer information, and article details. By leveraging data science techniques, we strive to uncover insights for effective product recommendations, enhance the shopping experience, reduce returns, and support sustainability. Our goal is to improve customer satisfaction and engagement through personalized recommendations.

## Executive Summary

### Problem
Customers often face difficulty finding suitable fashion items that match their preferences and needs, leading to a suboptimal shopping experience. Additionally, customers are not always exposed to the full catalog of available items, limiting their choices and potentially missing out on products that might interest them.

### Solution
To address these issues, we have developed personalized recommender systems. By analyzing customer preferences and purchasing patterns, the systems provide more accurate and tailored product recommendations. Furthermore, the systems increase exposure to a broader variety of items, helping customers discover products they might not have found otherwise.

### Impact
The implementation of these personalized recommender systems is expected to significantly enhance the shopping experience, leading to increased customer satisfaction and engagement. By providing better-suited recommendations and exposing customers to a wider range of products, the systems can help reduce the number of returns and contribute to more sustainable shopping practices.

## Process Overview
**Data Exploration and Preprocessing**
- **Data Exploration:** Conduct a comprehensive analysis of the dataset to identify trends, correlations, and key features that influence recommendations.
- **Data Cleaning:** Address missing values, remove redundant columns, and standardize data formats to ensure consistency and accuracy.
- **Feature Engineering:** Utilize Natural Language Processing (NLP) techniques to enrich item descriptions, extracting meaningful features for better model performance.

**Model Development**
- **Recommender Systems Creation:** Develop and iterate on three distinct recommender systems, each leveraging different algorithms and techniques:
  - **User-Item Matrix Recommender System:** Utilize collaborative filtering to recommend products based on user interaction history.
  - **Matrix Factorization Recommender System:** Implement matrix factorization techniques to uncover latent factors and improve recommendation accuracy.
  - **BERT-Based Recommender System:** Apply BERT for contextual understanding and semantic analysis of item descriptions, enhancing recommendation relevance.

**Evaluation and Prototyping**
- **Prototyping:** Develop initial prototypes for each recommender system to facilitate early-stage testing and iterative refinement. These prototypes help in identifying potential issues and areas for improvement.
- **System Evaluation:**
    - **Matrix Factorization Recommender System:** Conduct comprehensive evaluations
    - **User-Item Matrix and BERT Recommender Systems:** Perform qualitative assessments by analyzing individual recommendations.
-**Iterative Refinement:** Continuously refine the models based on evaluation results and user feedback.

## Organization

### Repository Structure

- **environment_data_links/**: Stores the `hm_env.yml` for setting up the environment and `data-links.md` includes instructions for accessing the data files and images.
- **notebooks/**: Includes all final Jupyter notebooks used in the project.
- **presentations/**: Contains presentations summarizing the project.
- **.gitignore**: Lists files and folders to be ignored by Git version control.
- **README.md**: Project landing page (this file).

## Datasets

- **Links to Datasets and Images:** The datasets and images required for this project are stored in publicly accessible cloud storage. Refer to the data-links.md file in the environment_data_links/ folder for the links and instructions to download the datasets.

## Environment Setup

To replicate the project environment, follow these steps:

1. **Download the Conda environment file**: You can download the environment YAML file from the `environment_data_links` folder, it is stored as `hm_env.yml`.
2. **Install the Conda environment**: Ensure you have Conda installed. Then, run:
    ```bash
    conda env create -f path_to_downloaded_hm_env.yml
    conda activate hm_env
    ```
**Note:** The BERT model notebooks were developed and optimized for Google Colab. For the best performance and compatibility, it is recommended to run those on Google Colab rather than Jupyter Notebooks or VS Code.
