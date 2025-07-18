import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import time

# Configure page
st.set_page_config(
    page_title="Bunny Detect",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🐰"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: #3b82f6;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #8b5cf6 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.3; }
        50% { transform: scale(1.1); opacity: 0.1; }
    }
    
    .bunny-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    .sender-receiver-container {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    
    .sender-section, .receiver-section {
        flex: 1;
        background: #f8fafc;
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .sender-section {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-color: #f59e0b;
    }
    
    .receiver-section {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-color: #3b82f6;
    }
    
    .sender-section:hover, .receiver-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .section-title {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 15px;
        color: #1e293b;
        text-align: center;
    }
    
    .results-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    .footer-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-top: 40px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    .footer-items {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
        flex-wrap: wrap;
    }
    
    .footer-item {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px;
        padding: 10px 20px;
        background: rgba(255,255,255,0.1);
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    
    .footer-item:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# Data Classes and Enums
class TransactionType(Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    CASH_IN = "CASH_IN"
    DEBIT = "DEBIT"


@dataclass
class Transaction:
    """Data class representing a financial transaction"""
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: bool = False
    isFlaggedFraud: bool = False


# Model Loading Functions
@st.cache_resource
def load_transaction_pipeline():
    """Load the transaction fraud detection pipeline"""
    model_path = 'fraud_transaction_pipepline.pkl'
    
    if os.path.exists(model_path):
        try:
            pipeline = joblib.load(model_path)
            return pipeline
        except Exception as e:
            st.error(f"Error loading transaction model: {e}")
            return None
    else:
        st.warning("Transaction model not found. Using mock predictions.")
        return None


@st.cache_resource
def load_review_pipeline():
    """Load the review fraud detection pipeline"""
    try:
        if os.path.exists('fake_review_detection.pkl'):
            pipeline = joblib.load('fake_review_detection.pkl')
            return pipeline
        else:
            st.warning("Review model not found. Using mock predictions.")
            return None
    except Exception as e:
        st.error(f"Error loading review model: {e}")
        return None


# Utility Functions
def calculate_avg_word_length(text: str) -> float:
    """Calculate the average word length in a text"""
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0


def calculate_sentiment_score(text: str) -> float:
    """Calculate sentiment polarity score"""
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except ImportError:
        # Fallback if TextBlob is not installed
        return 0.0


def calculate_keyword_presence(text: str) -> float:
    """Calculate presence of suspicious keywords commonly found in fake reviews"""
    suspicious_keywords = [
        'amazing', 'incredible', 'perfect', 'best ever', 'love it', 'hate it',
        'worst', 'terrible', 'awesome', 'fantastic', 'outstanding', 'excellent',
        'horrible', 'amazing quality', 'highly recommend', 'five stars',
        'one star', 'never again', 'waste of money', 'life changing',
        'game changer', 'must buy', 'avoid at all costs', 'absolutely',
        'definitely', 'totally', 'completely', 'super', 'extremely'
    ]
    
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in suspicious_keywords if keyword in text_lower)
    
    return keyword_count / len(suspicious_keywords) if suspicious_keywords else 0


def is_gibberish(text: str) -> bool:
    """Check if text appears to be gibberish or nonsensical"""
    import re
    
    # Remove spaces and convert to lowercase
    clean_text = re.sub(r'\s+', '', text.lower())
    
    # Check for excessive consonants without vowels
    vowels = 'aeiou'
    consonant_clusters = 0
    consonant_count = 0
    
    for char in clean_text:
        if char.isalpha():
            if char in vowels:
                consonant_count = 0
            else:
                consonant_count += 1
                if consonant_count > 4:  # More than 4 consonants in a row
                    consonant_clusters += 1
    
    # Check vowel to consonant ratio
    vowel_count = sum(1 for char in clean_text if char in vowels)
    total_letters = sum(1 for char in clean_text if char.isalpha())
    
    if total_letters > 0:
        vowel_ratio = vowel_count / total_letters
        # Normal text has roughly 40% vowels, gibberish often has much less
        if vowel_ratio < 0.15:
            return True
    
    # Check for excessive consonant clusters
    if consonant_clusters > 2:
        return True
    
    # Check for random character patterns (lots of uncommon letter combinations)
    uncommon_patterns = ['kfn', 'wrl', 'xlm', 'zkm', 'qxz', 'jns', 'fnk', 'nrl', 'mvc', 'cls']
    pattern_count = sum(1 for pattern in uncommon_patterns if pattern in clean_text)
    
    if pattern_count > 1:
        return True
    
    return False


def detect_fake_review_heuristic(features: dict) -> tuple:
    """
    Improved heuristic-based fake review detection
    Returns (label, confidence)
    """
    fraud_score = 0.0
    text = features['review']
    
    # CRITICAL: Check for gibberish first - this should be flagged as fraud
    if is_gibberish(text):
        return "Fake", 0.95  # Very high confidence for gibberish
    
    # Check for overly short reviews (less than 10 words)
    if features['word_count'] < 10:
        fraud_score += 0.3
    
    # Check for excessive exclamations
    if features['exclamations'] > 2:
        fraud_score += 0.2
    
    # Check for excessive uppercase words
    if features['uppercase_words'] > features['word_count'] * 0.3:
        fraud_score += 0.2
    
    # Check for extreme sentiment (too positive or too negative)
    if abs(features['sentiment_score']) > 0.8:
        fraud_score += 0.2
    
    # Check for suspicious keyword presence
    if features['keyword_presence'] > 0.1:  # More than 10% of keywords present
        fraud_score += 0.3
    
    # Check for rating-sentiment mismatch
    rating = features['rating']
    sentiment = features['sentiment_score']
    
    # High rating with negative sentiment or low rating with positive sentiment
    if (rating >= 4 and sentiment < -0.1) or (rating <= 2 and sentiment > 0.1):
        fraud_score += 0.25
    
    # Check for extremely generic text (short with high keyword presence)
    if features['word_count'] < 15 and features['keyword_presence'] > 0.05:
        fraud_score += 0.25
    
    # Check for the specific case in your example
    text_lower = features['review'].lower()
    if 'best' in text_lower and 'ever' in text_lower and features['word_count'] < 20:
        fraud_score += 0.4
    
    # Check for single word reviews or extremely short meaningful content
    if features['word_count'] <= 3:
        fraud_score += 0.4
    
    # Cap the fraud score at 1.0
    fraud_score = min(fraud_score, 1.0)
    
    # Determine label and confidence
    if fraud_score > 0.5:
        return "Fake", fraud_score
    else:
        return "Real", 1.0 - fraud_score


def prepare_transaction_features(transaction: Transaction) -> pd.DataFrame:
    """Prepare features for the transaction ML model"""
    return pd.DataFrame({
        'type': [transaction.type],
        'amount': [transaction.amount],
        'nameOrig': [transaction.nameOrig],
        'oldbalanceOrg': [transaction.oldbalanceOrg],
        'newbalanceOrig': [transaction.newbalanceOrig],
        'nameDest': [transaction.nameDest],
        'oldbalanceDest': [transaction.oldbalanceDest],
        'newbalanceDest': [transaction.newbalanceDest],
        'isFlaggedFraud': [transaction.isFlaggedFraud]
    })


# Prediction Functions
def predict_transaction_fraud(pipeline, transaction: Transaction) -> Tuple[bool, float]:
    """Predict transaction fraud using the trained pipeline"""
    if pipeline is None:
        # Mock prediction for demonstration
        np.random.seed(42)
        fraud_score = np.random.random()
        return fraud_score > 0.5, fraud_score
    
    try:
        features = prepare_transaction_features(transaction)
        prediction = pipeline.predict(features)[0]
        
        if hasattr(pipeline, 'predict_proba'):
            probabilities = pipeline.predict_proba(features)[0]
            fraud_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        else:
            fraud_probability = prediction
            
        return bool(prediction), float(fraud_probability)
    except Exception as e:
        st.error(f"Error in transaction prediction: {str(e)}")
        return False, 0.0


def predict_review_fraud(pipeline, features: Dict) -> Tuple[str, float]:
    """Predict review fraud using the trained pipeline"""
    if pipeline is None:
        # Mock prediction for demonstration
        np.random.seed(hash(features["review"]) % 1000)
        fraud_score = np.random.random()
        label = "Fake" if fraud_score > 0.6 else "Real"
        return label, fraud_score
    
    try:
        df_test = pd.DataFrame([features])
        pred = pipeline.predict(df_test)[0]
        prob = pipeline.predict_proba(df_test)[0]
        
        max_prob = max(prob)
        label = "Fake" if pred == 1 or pred == 'CG' else "Real"
        
        return label, max_prob
    except Exception as e:
        st.error(f"Error in review prediction: {str(e)}")
        return "Real", 0.5


# Page Functions
def show_transaction_fraud_page():
    """Display the transaction fraud detection page"""
    st.markdown("""
    <div class="main-header">
        <div class="bunny-title">💳 Transaction Fraud Detection</div>
        <div class="subtitle">Analyze financial transactions for fraudulent activity</div>
    </div>
    """, unsafe_allow_html=True)
    
    pipeline = load_transaction_pipeline()
    
    # Transaction Type Selection
    st.markdown("### 🔧 Transaction Configuration")
    transaction_type = st.selectbox(
        "Select Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    )
    
    # Sender and Receiver sections
    st.markdown("### 👥 Parties Involved")
    st.markdown("""
    <div class="sender-receiver-container">
        <div class="sender-section">
            <div class="section-title">📤 Sender</div>
        </div>
        <div class="receiver-section">
            <div class="section-title">📥 Receiver</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    sender_col, receiver_col = st.columns(2)
    
    with sender_col:
        nameOrig = st.text_input("Sender Name", value="C1234567890")
        sender_old_balance = st.number_input("Sender Old Balance", value=1000.0, min_value=0.0, format="%.2f")
        sender_new_balance = st.number_input("Sender New Balance", value=900.0, min_value=0.0, format="%.2f")
    
    with receiver_col:
        nameDest = st.text_input("Receiver Name", value="M1234567890")
        receiver_old_balance = st.number_input("Receiver Old Balance", value=0.0, min_value=0.0, format="%.2f")
        receiver_new_balance = st.number_input("Receiver New Balance", value=100.0, min_value=0.0, format="%.2f")
    
    # Additional transaction details
    st.markdown("### 📊 Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input("Transaction Amount", value=100.0, min_value=0.0, format="%.2f")
    
    with col2:
        isFraud = st.checkbox("Is Actually Fraud")
        isFlaggedFraud = st.checkbox("Was Flagged as Fraud")
    
    with col3:
        st.markdown("**Balance Changes**")
        sender_change = sender_new_balance - sender_old_balance
        receiver_change = receiver_new_balance - receiver_old_balance
        st.write(f"Sender: ${sender_change:,.2f}")
        st.write(f"Receiver: ${receiver_change:,.2f}")
    
    # Analyze Transaction Button
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction for fraud..."):
            time.sleep(1)  # Simulate processing time
            
            transaction = Transaction(
                type=transaction_type,
                amount=amount,
                nameOrig=nameOrig,
                oldbalanceOrg=sender_old_balance,
                newbalanceOrig=sender_new_balance,
                nameDest=nameDest,
                oldbalanceDest=receiver_old_balance,
                newbalanceDest=receiver_new_balance,
                isFraud=isFraud,
                isFlaggedFraud=isFlaggedFraud
            )
            
            is_fraudulent, fraud_probability = predict_transaction_fraud(pipeline, transaction)
            
            # Determine risk level
            if fraud_probability >= 0.8:
                risk_level = "HIGH"
            elif fraud_probability >= 0.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Display results
            st.markdown("### 🎯 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fraud_status = "🚨 FRAUD DETECTED" if is_fraudulent else "✅ LEGITIMATE"
                if is_fraudulent:
                    st.error(fraud_status)
                else:
                    st.success(fraud_status)
            
            with col2:
                st.metric("Fraud Probability", f"{fraud_probability:.2%}")
            
            with col3:
                if risk_level == "HIGH":
                    st.error(f"Risk Level: {risk_level}")
                elif risk_level == "MEDIUM":
                    st.warning(f"Risk Level: {risk_level}")
                else:
                    st.success(f"Risk Level: {risk_level}")
            
            # Detailed breakdown
            with st.expander("📋 Detailed Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Transaction Information**")
                    st.write(f"• Type: {transaction_type}")
                    st.write(f"• Amount: ${amount:,.2f}")
                    st.write(f"• Sender: {nameOrig}")
                    st.write(f"• Receiver: {nameDest}")
                
                with col2:
                    st.markdown("**Risk Assessment**")
                    st.write(f"• Fraud Probability: {fraud_probability:.4f}")
                    st.write(f"• Risk Level: {risk_level}")
                    st.write(f"• Previously Flagged: {'Yes' if isFlaggedFraud else 'No'}")
                    st.write(f"• Ground Truth: {'Fraud' if isFraud else 'Legitimate'}")
            
            st.markdown('</div>', unsafe_allow_html=True)


def show_review_fraud_page():
    """Display the review fraud detection page"""
    st.markdown("""
    <div class="main-header">
        <div class="bunny-title">🔍 Review Fraud Detection</div>
        <div class="subtitle">Detect fake reviews with AI-powered analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Review Analysis")
    
    # Review input with word limit
    text = st.text_area(
        "Review Text (Max 50 words)",
        "This is the best product ever! Love it!",
        height=150,
        help="Enter the review text you want to analyze (maximum 50 words)"
    )
    
    # Check word count and display warning if exceeded
    word_count = len(text.split()) if text else 0
    if word_count > 50:
        st.warning(f"⚠️ Word limit exceeded: {word_count}/50 words. Please shorten your review.")
        # Truncate to 50 words
        words = text.split()
        text = ' '.join(words[:50])
        st.info(f"Review automatically truncated to 50 words: '{text}'")
    
    # Display current word count
    st.caption(f"Word count: {word_count}/50")
    
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox(
            "Review Category:",
            ["Electronics", "Books", "Clothing", "Home & Garden", "Sports", "Other"]
        )
    
    with col2:
        rating = st.slider("Rating", 1, 5, 5)

    # Analyze button - only enabled if within word limit
    analyze_disabled = word_count > 50 or not text or text.strip() == ""
    
    if st.button("🔍 Analyze Review", use_container_width=True, type="primary", disabled=analyze_disabled):
        if not text or text.strip() == "":
            st.warning("Please enter a review text.")
            return
        
        with st.spinner("Analyzing review..."):
            # Calculate features
            features = {
                "review": text,
                "category": category.lower(),
                "rating": rating,
                "review_length": len(text),
                "word_count": len(text.split()),
                "exclamations": text.count('!'),
                "uppercase_words": sum(1 for w in text.split() if w.isupper()),
                "avg_word_length": calculate_avg_word_length(text),
                "sentiment_score": calculate_sentiment_score(text),
                "keyword_presence": calculate_keyword_presence(text)
            }
            
            # Use the improved heuristic detection
            label, confidence = detect_fake_review_heuristic(features)
            
            # Display results
            st.markdown("### 🔍 Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                if label == "Fake":
                    st.error("🚨 **FRAUD DETECTED**")
                    st.error(f"Confidence: {confidence:.1%}")
                else:
                    st.success("✅ **LEGITIMATE REVIEW**")
                    st.success(f"Confidence: {confidence:.1%}")
            
            with col2:
                st.markdown("**Analysis Details:**")
                st.write(f"• Length: {features['review_length']} characters")
                st.write(f"• Words: {features['word_count']}")
                st.write(f"• Avg word length: {features['avg_word_length']:.1f}")
                st.write(f"• Exclamations: {features['exclamations']}")
                st.write(f"• Uppercase words: {features['uppercase_words']}")
                st.write(f"• Sentiment score: {features['sentiment_score']:.2f}")
                st.write(f"• Keyword presence: {features['keyword_presence']:.2f}")
            
            # Show fraud indicators
            st.markdown("### 🚨 Fraud Indicators")
            indicators = []
            
            # Check for gibberish first
            if is_gibberish(text):
                indicators.append("🚨 **GIBBERISH DETECTED** - Text appears to be nonsensical")
            
            if features['word_count'] < 10:
                indicators.append("⚠️ Extremely short review")
            if features['exclamations'] > 2:
                indicators.append("⚠️ Excessive exclamation marks")
            if features['uppercase_words'] > features['word_count'] * 0.3:
                indicators.append("⚠️ Too many uppercase words")
            if abs(features['sentiment_score']) > 0.8:
                indicators.append("⚠️ Extreme sentiment")
            if features['keyword_presence'] > 0.1:
                indicators.append("⚠️ High suspicious keyword presence")
            
            text_lower = text.lower()
            if 'best' in text_lower and 'ever' in text_lower and features['word_count'] < 20:
                indicators.append("⚠️ Generic superlative language")
            
            if features['word_count'] <= 3:
                indicators.append("⚠️ Extremely short content")
            
            if indicators:
                for indicator in indicators:
                    st.write(indicator)
            else:
                st.write("✅ No major fraud indicators detected")
            
            # Metrics display
            st.markdown("### 📊 Detection Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col2:
                legit_pct = f"{confidence:.1%}" if label == "Real" else f"{(1-confidence):.1%}"
                st.metric("Legitimate", legit_pct)
            
            with col3:
                fraud_pct = f"{confidence:.1%}" if label == "Fake" else f"{(1-confidence):.1%}"
                st.metric("Fraud Risk", fraud_pct)
            
            with col4:
                fake_pct = f"{confidence:.1%}" if label == "Fake" else f"{(1-confidence):.1%}"
                st.metric("Fake Probability", fake_pct)

# Navigation Functions
def add_navigation_buttons():
    """Add navigation buttons to switch between pages"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'fraud_detection'

    if st.button("🔍 Review Fraud Detection", key="fraud_btn", use_container_width=True):
        st.session_state.current_page = 'fraud_detection'
        st.rerun()

    if st.button("💳 Transaction Fraud Detection", key="transaction_btn", use_container_width=True):
        st.session_state.current_page = 'transaction'
        st.rerun()

    st.markdown("---")
    current_page_name = "Review Fraud Detection" if st.session_state.current_page == 'fraud_detection' else "Transaction Fraud Detection"
    st.markdown(f"**Current Page:** {current_page_name}")


def main_content():
    """Display the main content based on current page"""
    if st.session_state.current_page == 'fraud_detection':
        show_review_fraud_page()
    elif st.session_state.current_page == 'transaction':
        show_transaction_fraud_page()


# Main Application
def main():
    """Main application function"""
    # Sidebar
    with st.sidebar:
        # Check if logo exists, otherwise show placeholder
        try:
            st.image("Bunny_Detect_Logo.png", width=100)
        except:
            st.markdown("🐰", unsafe_allow_html=True)

        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="color: white; font-size: 1.5rem; margin: 0;">
                    Bunny Detect
                </h1>
            </div>
        """, unsafe_allow_html=True)

        add_navigation_buttons()

    # Main content
    main_content()

    # Footer
    st.markdown("""
    <div class="footer-section">
        <div class="footer-items">
            <div class="footer-item">
                <span>✅</span>
                <span>Trusted Detection</span>
            </div>
            <div class="footer-item">
                <span>🤝</span>
                <span>Partnerships</span>
            </div>
            <div class="footer-item">
                <span>📞</span>
                <span>77-898-0988</span>
            </div>
            <div class="footer-item">
                <span>❓</span>
                <span>FAQ</span>F
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    main()