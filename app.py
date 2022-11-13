from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext
import pyspark.sql.functions as F
import pyspark.sql.types as T
import streamlit as st
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner"])
car=[",","\"","\'",".","!","?",":","(",")"]  # ici on filtre toute la ponctuation
def clean_text(text):
  doc = nlp(text)
  tab_result = []
  for token in doc :
    if token.is_stop == False :
      if(token.lemma_.lower() not in car ):
        tab_result.append(token.lemma_.lower())
  #Ne garder que les LEMME et enlever les STOPWORDS
  return tab_result
df_clean_text = spark.read.parquet("clean/first_clean.pqt")
df_clean_text.show()

#Implementing the word2Vec model
word2Vec = Word2Vec(vectorSize=300, seed=42, inputCol="words",       
outputCol="features")
w2vmodel = word2Vec.fit(df_clean_text)
#Nouvelle base de données avec les vecteurs
w2vdf = w2vmodel.transform(df_clean_text)

# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="words", outputCol="features", minDF=10.0)

model = cv.fit(df_clean_text)

resultCV = model.transform(df_clean_text)


def LR(data):
    result_cast=data.withColumn("toxic",data.toxic.cast('int'))
    # Load training data
    train, test = result_cast.randomSplit([0.8, 0.2], seed=12345)
    train
    lr = LogisticRegression(regParam=0.3, featuresCol="features",labelCol='toxic')
    lr.setMaxIter(10)
    lr.setRegParam(0.1)
    # Fit the model
    model = lr.fit(train)
    return model


def prediction(txt):
    d = [{'comment_text': str(txt)}]
    df = spark.createDataFrame(d)
    #Cleaning avec spacy
    clean_text(txt)
    #Ajout de la nouvelle colonne 'col' avec le commentaire nettoyé (clean_comm) dans notre base de données 'df'
    udf_clean_text = F.udf(clean_text, T.StringType())
    df_clean_text = df.withColumn("clean_comment_text", udf_clean_text("comment_text"))

    # Tokenisation
    tokenizer = Tokenizer(inputCol="clean_comment_text", outputCol="tokenised_comment")
    tokenizedData = tokenizer.transform(df_clean_text)
    tokenizedData.show(n=5)
    #Implementing the word2Vec model
    word2Vec = Word2Vec(minCount=0, vectorSize=300, seed=42, inputCol="tokenised_comment",
                        outputCol="features")
    w2vmodel = word2Vec.fit(tokenizedData)
    w2vdf = w2vmodel.transform(tokenizedData)
    return w2vdf
    



if __name__=="__main__":
    st.title("Text Mining Project")
    vect = st.selectbox(
     'Veuillez choisir la vectorisation',
     ('CountVectorizer', 'Word2Vec', 'TF-IDF'))
     
    modele = st.selectbox(
     'Veuillez choisir le modèle',
     ('Régression Logistique', 'Random Forest', 'SVM'))
    topic= st.text_input("Veuillez saisir la phrase : ", "")
    paragraphe=st.empty()
    if vect=="CountVectorizer" and modele=="Régression Logistique":
    
        if topic:
            model=LR(resultCV)
            print(type(model))
            s=prediction(topic)
            predictions = model.transform(s)
            predictions.show()
            paragraphe.markdown(predictions.select('prediction').show())
