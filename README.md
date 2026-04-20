# Problem Definition

This project focuses on cross-sectional factor machine learning, aiming to predict the forward 21-trading-day return of selected financial assets. The target variable is the forward return over a horizon of 21 days, using a no look-ahead policy to ensure robustness in predictions.

## How to Run Locally

To run this project locally, ensure you have the necessary dependencies listed in `requirements.txt`. Set up your environment variables as outlined in `.env.example`, and use the provided scripts to execute the end-to-end training and evaluation pipeline.

Additionally, artifacts will be uploaded to a Supabase Storage bucket via a service role key for easier management and access.