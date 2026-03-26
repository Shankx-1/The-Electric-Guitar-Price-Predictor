# Electric Guitar Price Predictor (AI/ML BYOP)

**Fundamentals of AI and ML - Evaluated Course Project**
**Student:** Harman Singh | **Reg No:** 25BCE10077

## Problem Statement
Buying or selling a second-hand electric guitar can be confusing. Prices vary wildly based on the brand, the type of pickups, the age of the instrument, and its overall condition. Buyers often overpay, and sellers often undervalue their gear. 

## The Solution
This project uses Machine Learning to predict the fair market price of an electric guitar based on its specifications. By utilizing a **Random Forest Regressor**, the model learns the complex relationships between brand prestige, hardware (pickups), wear-and-tear, and the final price.

## Tech Stack
* **Language:** Python
* **Libraries:** `pandas` (Data manipulation), `scikit-learn` (Machine Learning modeling), `numpy` (Numerical operations)

## How to Run This Project
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn`
3. Run the python script: `python guitar_price_predictor.py`
4. The script will automatically load the dataset, preprocess the text data using One-Hot Encoding, train the Random Forest model, and output the predicted prices versus the actual prices.

## Key Learnings
This project demonstrates data preprocessing (converting categorical text to numerical data), splitting datasets into training and testing sets, and evaluating a regression model using Mean Absolute Error (MAE).
