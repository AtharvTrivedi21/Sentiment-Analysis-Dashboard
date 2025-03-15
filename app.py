import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
from transformers import pipeline

nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize sentiment analyzers
sia = SentimentIntensityAnalyzer()
sentiment_pipeline = pipeline('sentiment-analysis')

# Streamlit App Configuration
st.set_page_config(page_title='Sentiment Analysis Dashboard', layout='wide')
st.sidebar.title('Sentiment Analysis Dashboard')
st.sidebar.write('Analyze customer sentiments from text data - real-time or batch processing!')

# Sidebar for Analysis Type and Model Selection
analysis_type = st.sidebar.radio('Choose Analysis Type:', ['Single Text Analysis', 'Batch Analysis'])
model_choice = st.sidebar.selectbox('Choose Sentiment Analysis Model:', ['VADER', 'TextBlob', 'BERT-based Transformers'])

if analysis_type == 'Single Text Analysis':
    st.header('🔹 Single Text Sentiment Analysis')
    user_input = st.text_area('Enter text for sentiment analysis:', '')

    if st.button('Submit', key='single_submit') and user_input:
        if model_choice == 'VADER':
            sentiment = sia.polarity_scores(user_input)
            st.subheader('VADER Sentiment Analysis Results')
            st.write(f"**Negative:** {sentiment['neg']*100:.2f}% | **Neutral:** {sentiment['neu']*100:.2f}% | **Positive:** {sentiment['pos']*100:.2f}%")
            compound = sentiment['compound']
            sentiment_label = 'Positive 😊' if compound >= 0.05 else 'Negative 😡' if compound <= -0.05 else 'Neutral 😐'
            st.write(f'Overall Sentiment: **{sentiment_label}**')

        elif model_choice == 'TextBlob':
            polarity = TextBlob(user_input).sentiment.polarity
            st.subheader('TextBlob Sentiment Polarity')
            st.write(f'Polarity Score: {polarity:.2f}')
            sentiment_label = 'Positive 😊' if polarity > 0 else 'Negative 😡' if polarity < 0 else 'Neutral 😐'
            st.write(f'Overall Sentiment: **{sentiment_label}**')

        else:
            transformer_result = sentiment_pipeline(user_input)[0]
            st.subheader('BERT-based Sentiment Analysis')
            st.write(f"Label: {transformer_result['label']}, Score: {transformer_result['score']:.2f}")

elif analysis_type == 'Batch Analysis':
    st.header('🔹 Batch Sentiment Analysis (CSV Upload)')
    uploaded_file = st.file_uploader('Upload a CSV file with a "text" column:', type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'text' in df.columns:
            if model_choice == 'VADER':
                df['Sentiment_Score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
            elif model_choice == 'TextBlob':
                df['Sentiment_Score'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
            else:
                df['Sentiment_Score'] = df['text'].apply(lambda x: sentiment_pipeline(x)[0]['score'] if sentiment_pipeline(x)[0]['label'] == 'POSITIVE' else -sentiment_pipeline(x)[0]['score'])

            # Categorize sentiments
            df['Sentiment'] = np.where(df['Sentiment_Score'] > 0.05, 'Positive',
                                       np.where(df['Sentiment_Score'] < -0.05, 'Negative', 'Neutral'))

            st.subheader('Data Preview:')
            st.dataframe(df.head())

            # Distribution Plot
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader('Sentiment Distribution')
            with col2:
                st.subheader('Sentiment Proportion')

            # st.subheader('Sentiment Distribution vs Sentiment Proportion')

            col3, col4 = st.columns(2)  # Creates two equal-width columns

            # Sentiment Distribution (Histogram)
            with col3:
                fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
                sns.histplot(df['Sentiment'], ax=ax, palette='coolwarm', kde=True)
                ax.set_xlabel('Sentiment')
                ax.set_ylabel('Count')
                plt.tight_layout()
                st.pyplot(fig)

            # Sentiment Proportion (Pie Chart)
            with col4:
                sentiment_counts = df['Sentiment'].value_counts()
                fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
                ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['red', 'green', 'yellow'])
                # ax.set_title('Sentiment Breakdown')
                plt.tight_layout()
                st.pyplot(fig)

            # Word Cloud
            st.subheader('Word Cloud of Text')
            # col5, col6 = st.columns([1, 1])

            # with col5:
            all_text = ' '.join(df['text'].dropna())
            wordcloud = WordCloud(background_color='white', width=1200, height=600).generate(all_text)

            fig, ax = plt.subplots(figsize=(12, 6), dpi=200)  # Increased size
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)

            
        else:
            st.error('The CSV must contain a "text" column.')

st.sidebar.write('Built with Streamlit, NLTK, TextBlob, and Transformers!')
