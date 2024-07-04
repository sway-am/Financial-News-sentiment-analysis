# Financial-News-sentiment-analysis
https://huggingface.co/spaces/sway0604/news_sentiment

**SUMMARY**
<pre>
  AIM : To develop a transformer based model to analyse financial news sentiment either by direct text input or by scraping the headline via web link.
  TECHNOLOGIES USED : Hugging face, Stream lit, Distill BERT, newspaper3k.
  DATASET : Financial Phrase Bank.
</pre>


**APPROACH**

<pre>
  1. Distill BERT pretrained model was fine tuned on the Financial Phrase Bank dataset.
  2. It was able to achieve an accuracy score of 80.79%, precision of 0.81 and F1 score of 0.77.
  3. newspaper3k library was used to scrape the headline of the website, the model was deployed on Hugging face via streamlit.
</pre>

  

  

