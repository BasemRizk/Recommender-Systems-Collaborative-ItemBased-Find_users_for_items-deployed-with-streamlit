import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
import warnings; warnings.simplefilter('ignore')

full_df=pd.read_csv('full_dataset.csv')
full_df['CustomerID'] = full_df['CustomerID'].astype(int)
full_df['InvoiceDate']=pd.to_datetime(full_df['InvoiceDate'])
full_df['InvoiceDate'] = full_df['InvoiceDate'].dt.strftime('%d-%m-%Y')
df=full_df.copy()


df['Desc']=df['Description']
df = df.groupby(['StockCode']).agg({'Quantity': 'sum', 'UnitPrice': 'first','Description':'first','Desc':'first','InvoiceNo':'count'})
df = df.reset_index()
df=df[df['Quantity']>0]
stemmer = SnowballStemmer('english')
df['Description']=df['Description'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))


ratings=pd.read_csv('product_rating.csv')
from surprise import Dataset, NormalPredictor, Reader
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['CustomerID', 'StockCode', 'purchase_rate']], reader)
svd=SVD()
svd.n_epochs=10
cross_validate( svd,data, measures=['RMSE'], cv=5)
trainset = data.build_full_trainset()
svd.fit(trainset)



tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['Description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
df = df.reset_index()
indices = pd.Series(df.index, index=df['StockCode'])

       
def get_scores(title):
    results=[]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    for score in(sim_scores):
        if(score[1]>0):
            results.append(score)
    if(len(results))<1:
        return 'No Such elements like this'
    item_indices = [i[0] for i in results]    
    scores = [i[1] for i in results]    
    res=df.loc[item_indices,['StockCode','Desc','UnitPrice']]
    res['score']=scores
    res=res[res['score']>0.2]

    return res
def fav_clients(title):
    df=get_scores(title)
    df_clients=pd.DataFrame()
    for item in df['StockCode']:
        y=full_df[full_df['StockCode']==item]
        df_clients=pd.concat([df_clients, y], axis=0)
    merged_df = pd.merge(df_clients, df, on='StockCode')
    merged_df['rate']=merged_df['Quantity']*merged_df['score']
    customer_totals = merged_df.groupby('CustomerID')['rate'].sum()
    top_customers = customer_totals.sort_values(ascending=False).head(5)
    return top_customers
def colab_filter(user,title):
    res=get_scores(title)
    ests=[]
    for item in res['StockCode']:
        est=svd.predict(user, item).est
        ests.append(est)
    res['rates']=ests
    res = res.sort_values('rates', ascending=False)
    res=res[['StockCode','Desc','UnitPrice']]
    return res.head()