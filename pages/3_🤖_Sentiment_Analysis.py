import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="🤖 Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="✨"
)


st.sidebar.image('Source\sentiment-analysis.png', caption='Sentiment & Emotion', width=150, use_column_width=False)

# ================================================================================================================


# Menampilkan judul dengan dekorasi yang terpusat
st.markdown(
    """
    <div style="background-color:#333; padding:15px;">
        <h1 style='text-align: center;'>
            <span style='display: block;font-size: 75px; color: white'>
                🤖<strong>Sentiment Analysis</strong>🤖
            </span>
        </h1>
            <span style='display: block; font-size: 30px; color: white; text-align: center;'>
                Positive | Negative | Neutral
            </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ================================================================================================================


from transformers import pipeline

indonesian_roberta_base_sentiment_classifier = pipeline(
    model="w11wo/indonesian-roberta-base-sentiment-classifier", 
    return_all_scores=True
)

# ================================================================================================================
# Import necessary libraries
import streamlit as st

# Streamlit layout
st.title("Let's Try Guys!")

# Input teks untuk prediksi
with st.container():
    text_input_col, result_col = st.columns([2, 1])  # Define column widths

    with text_input_col:
        text_input = st.text_area("Masukkan teks untuk analisis sentiment:")

    # Fungsi untuk melakukan prediksi sentimen
    def predict_sentiment(text):
        scores = indonesian_roberta_base_sentiment_classifier(text)
        # Flatten the nested list of dictionaries
        flattened_scores = [item for sublist in scores for item in sublist]
        # Create a DataFrame from the flattened scores
        df = pd.DataFrame(flattened_scores)
        # Rename the columns for clarity
        df.rename(columns={'label': 'Sentiment', 'score': 'Score'}, inplace=True)
        return df

    # Melakukan prediksi dan menampilkan hasil
    with result_col:
        if st.button("Predict"):
            if text_input:
                prediction = predict_sentiment(text_input)
                max_index = prediction['Score'].argmax()
                # Get the corresponding label
                max_label = prediction.iloc[max_index]["Sentiment"]
                max_score = prediction.iloc[max_index]["Score"]
                st.write("_Sentimen yang Diprediksi:_", f"**{max_label}**", "_dengan confidence score_", f"**{max_score * 100:.2f}%**")


                
# Display the prediction result outside of the two-column layout
if 'prediction' in locals():
    st.table(prediction)

# ================================================================================================================
def main():
    st.title("Dynamic Feature Addition")
    st.write("Sentiment analysis for text in a dataframe")
    # Button to add new feature
    if st.button("Try the Features"):
        
        try:
            # Dapatkan DataFrame dari session state
            df = st.session_state['key']
        except KeyError:
            df = None

        if df is None:
            st.write("Dataframe belum diinisialisasi.")
            return

        # Dapatkan DataFrame dari session state
        df = st.session_state['key']
        sampled_df = df.sample(n=5, random_state=70)
        st.text("5 Sample Data From Dataframe")
        st.table(sampled_df)
        # You can add more Streamlit elements here to display additional features
        # ==================================================================================
        
        # Fungsi untuk memprediksi sentimen menggunakan model sentimen bahasa Indonesia
        def predict_sentiment_score(text):
            scores = indonesian_roberta_base_sentiment_classifier(text)
            return scores
        
        predicted_sentiments = []
        # Iterasi melalui setiap baris DataFrame dan memprediksi sentimen
        for i, text in sampled_df['JAWABAN'].items():
            results = indonesian_roberta_base_sentiment_classifier(text)
            max_score_index = np.argmax([entry['score'] for entry in results[0]])
            predicted_sentiment = results[0][max_score_index]['label']
            predicted_sentiments.append(predicted_sentiment)

        sampled_df['Predicted_Sentiment'] = predicted_sentiments
        # Tampilkan DataFrame setelah prediksi sentimen
        st.table(sampled_df[['JAWABAN', 'Predicted_Sentiment']])

if __name__ == "__main__":
    main()

