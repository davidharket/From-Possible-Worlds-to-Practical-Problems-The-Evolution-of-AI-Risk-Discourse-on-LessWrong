# Code for Digital Methods Exam Analysis

This repository contains the core Python scripts used for the analysis conducted for our MSc thesis, "An Analysis of the AI Risk Discourse on LessWrong". The code is shared to provide full transparency on the methods used to arrive at our findings.

## Project Overview

The project analyzes posts from the website LessWrong to identify and track discussions related to AI risk. The analysis pipeline consists of four main stages, each represented by a key script in this repository:

1.  **Data Retrieval and Processing**: A consolidated script for collecting post metadata, filtering it based on specific tags (`ai risk` and `alignment`), and scraping the full text content for each relevant post and its comments.
2.  **Post Classification**: A suite of classifiers to categorize text segments as discussing "abstract" or "tangible" AI risks. The repository includes the primary similarity-based classifier used in our paper, as well as a script showcasing the alternative methods we explored.
3.  **Topic Modeling**: A script to perform topic modeling using BERTopic on the collected text data to identify latent themes in the discourse.
4.  **Quantitative Analysis**: A script for performing the final statistical analysis (t-tests, z-tests, effect sizes, etc.) on the classified data to generate the results presented in our appendix.

## Repository Structure

The repository is organized into the following directories, containing the key scripts that form our analytical pipeline.

*   `data_retrieval_and_processing/`: Contains the consolidated script for all data collection and processing.
    *   `master.py`: A three-stage pipeline script that (1) fetches all raw post metadata from LessWrong, (2) filters these posts for relevant tags, and (3) scrapes the full HTML content for the filtered posts and their comments.

*   `post_classification/`: Contains the primary classification script used in our analysis and a script representing our exploration of alternative methods.
    *   `cosine_sim_classification.py`: The final, robust classifier used in our paper. It uses sentence embeddings and cosine similarity with a comprehensive list of example sentences. It includes logic to handle long documents by processing them in chunks.
    *   `other_classifiers.py`: A consolidated script that showcases the other methods we explored for comparison, including a heuristic keyword-based approach, a Flan-T5 model, and a simple supervised ML model trained on heuristic labels.

*   `topic_modelling/`: Contains the primary script for topic modeling.
    *   `topic.py`: The core script for our topic modeling process. It loads the scraped text data, initializes and trains a BERTopic model, and saves the resulting topic information and topics-over-time data.

*   `quantiative_analysis/`: Contains the script for the final statistical analysis.
    *   `statistical_analysis.py`: This script takes the classified data and performs the statistical tests (z-tests, t-tests, confidence threshold analysis) needed to generate the summary tables presented in the appendix of our paper.

*   `immersion_journal/`: Contains supplementary material from our qualitative analysis.
    *   `immersion_journal.csv`: A log documenting our initial qualitative netnographic immersion into the LessWrong community, as described in the paper.

## How to Understand This Repository

This repository is intended to provide a clear view of our core methodology. A conceptual workflow to reproduce the analysis would be:

1.  **Run `data_retrieval_and_processing/master.py`** to collect and filter the data from LessWrong.
2.  **Run `post_classification/cosine_sim_classification.py`** on the output from the previous stage to classify the posts. The `other_classifiers.py` script can be reviewed to understand the alternative methods we compared against.
3.  **Run `topic_modelling/topic.py`** to generate the topic model from the collected data.
4.  **Run `quantiative_analysis/statistical_analysis.py`** on the output of the classification stage to generate the final statistical results.

## Important Notes

This repository is shared to provide full transparency into our analytical process. The scripts contained here are the result of merging and condensing a larger number of exploratory files used during our research.

*   **Consolidated Scripts**:  The files in this repository represent a systematic reconstruction of our analysis pipeline. Many smaller, single-task scripts were combined into the larger, sequential scripts you see here. This compression was done to showcase our logic clearly, but the process was somewhat unpolished.
*   **Purpose of Sharing**: The main purpose of this repository is to display the logic of our analysis strategy. It is not intended as a polished, one-click application, but rather as a reference for understanding how our results were generated.
*   **Focus on Core Logic**: To maintain clarity, some secondary analyses (e.g., specific visualization generation, author overlap analysis) have been omitted. The code present is a faithful representation of the core data collection, classification, and analysis pipeline.

We believe that sharing our code, even in this consolidated and unpolished state, is a valuable step towards transparency and reproducibility in digital methods research.