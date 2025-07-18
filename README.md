# HackAttack-Bunny
Fraud Detection Platform by HackAttack Bunny
This platform provides AI-powered fraud detection tools to help users identify suspicious reviews and transactions.

It consists of two modules:
- Fake Review Detection
- Transaction Fraud Detection

## Installation

Make sure Python is installed, then install the required libraries:
pip install streamlit 
pip install pandas 
pip install numpy 
pip install joblib 
pip install plotly 

## How does it Work
After the installation, users will be able to launch the Streamlit app with the command:
streamlit run Final_Bunny_Detect_Version.py

## Features: Fake Review Deection:
- Legitimate Check (Identifies whether the review is legitimate or fake review.)
- AI Confidence (Shows the confidence level of the legitimacy or fraud risk.)
- Fraud indicators (Used to detect the potential fraudulent)
- Detection metrics (Confidence %, Fraud risk %, Fake probability %)

How to used:
1. Users can input the product review into the Review Text box.
2. Select the product review category. 
3. Rate the product by sliding the slider from 1 to 5.
4. Click on the Analyse Review button to detect fraud and identify whether the review is fake or show the confidence level of the result.
5. The system will show the result: 
- Whether the review is legitimate or fake.
- Analysis details and metrics.
- Fraud risk and fake probability. 

## Features: Transaction Fraud Detection
- Transaction type selection.
- Parties involved.
- Real-time Balance Tracking (Since it auto calculates the transaction impact.)
- Fraud Analysis (AI evaluates transaction behaviour. Flags suspicious patterns based on balance discrepancies or irregular transfers)
- Trusted Detection: Indicates whether the transaction is flagged or verified

How to use:
1. Select the transaction type from the dropdown list. 
2. Enter sender and receiver details. (e.g. )
3. Enter the actual transaction amount.
4. Flag the transaction (Check whether it is fraud or was flagged as fraud)
5. Click on the Analyse Transaction button to access the inputs and return the fraud status.


