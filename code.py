import pandas as pd
import nltk
import string
# Read the text file into a DataFrame
dff = pd.read_csv("p.txt", delimiter='\t')  
print(dff.head())

#remval stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
"/".join(stopwords.words('english'))
stop_words=set(stopwords.words('english'))
dff = dff.applymap(lambda x: x.lower() if isinstance(x, str) else x)
print(dff.head(10))
print(stop_words)

def remove_stop(x):
    return " ".join([word for word in str(x).split() if word not in stop_words])

dff = dff.applymap(remove_stop)

dff.head(10)
from collections import Counter

# Step 1: Flatten the DataFrame into a single Series
series = dff.stack()

# Step 2: Tokenize the text in each cell to extract individual words
words = series.str.split(expand=True).stack()

# Step 3: Count the occurrences of each word
words_count = Counter(words)

print(words_count)
