#nlp.py 
# 1)task (tokenization, normalization, stop words, lemmatization, stemming
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

#Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Text normalization (lowercasing)
    text = text.lower()

    # Tokenization (splitting text into words)
    tokens = word_tokenize(text)
 
    # Removing stop words
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]

    # Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return {
        "original_text": text,
        "filtered_words": filtered_words,
        "stemmed_words": stemmed_words,
        "lemmatized_words": lemmatized_words
    }

your_text = "Groundwater stands as a crucial and renewable resource globally, playing a vital role in the Earth's natural water cycle. This valuable resource, found in underground layers, primarily originates from sources such as rainwater and water flowing in streams. In the past, the methods employed to ascertain the existence and potential of groundwater involved various time-intensive and expensive techniques. These included drilling, alongside geophysical, geological, hydrogeological, and geo-electrical methods. However, these traditional approaches often fell short in terms of providing comprehensive coveragRecent advancements have highlighted that the integration of remote sensing and Geographic Information System (GIS) technologies presents a more efficient way to map and evaluate groundwater potential. This study explores the effectiveness of a GIS-based approach, utilizing the Modified DRASTIC Model. This model integrates various critical factors that have a direct impact on the presence and behavior of groundwater. It considers several surface attributes that are indicative of groundwater potential. These attributes encompass aspects such as the geological characteristics of the area, the texture of the soil, prevalent land use patterns, lithology, the typology of landforms, the degree of slope steepness, the presence and characteristics of lineaments, as well as the nature of the drainage systems in the region. In order to predict changes in groundwater potential, this study employed the MOLUSCE tool, a sophisticated plugin used in QGIS. This tool applies a combination of advanced techniques including Artificial Neural Networks (ANN), multicriteria evaluation, weights of evidence, and Logistic Regression (LRs) algorithms to predict changes in land use and land cover. The reliability of these predictions was notably high, as evidenced by a kappa value of 0.83. The analysis yielded insightful results, particularly concerning different regions' groundwater potential. For instance, it was observed that areas in the Southwest region consistently exhibited low to very low groundwater potential over the years. In contrast, the central region displayed a high to very high potential consistently. Moreover, the changes in potential over the years in these regions were minimal.However, a notable prediction made by the study is that by the year 2042, the eastern region of Kiambu County is expected to experience a decrease in groundwater potential. This insight is particularly important for future planning and management of water resources in the area. The application of these advanced GIS and remote sensing techniques, therefore, not only provides a more efficient and comprehensive method for assessing groundwater potential but also offers valuable foresight for effective water resource management."

preprocessed = preprocess_text(your_text)

print("Original Text:", preprocessed['original_text'])
print("Filtered Words:", preprocessed['filtered_words'])
print("Stemmed Words:", preprocessed['stemmed_words'])
print("Lemmatized Words:", preprocessed['lemmatized_words'])

#text Analysis 
#based on the previous code (of cleaning) we are going to analyze the text 
# 2) Word Frequency Analysis (Counting how often each word appears in a text). 

from collections import Counter

#Count the frequency of each word in the filtered text
word_freq = Counter(preprocessed['filtered_words'])

print("Word Frequencies:", word_freq)

# 3) Sentiment Analysis (Determining if the sentiment of a text is positive, negative, or neutral).

from nltk.sentiment import SentimentIntensityAnalyzer

#Download the VADER lexicon
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

#Analyze the sentiment of the original text
sentiment = analyze_sentiment(your_text)

print("Sentiment Analysis:", sentiment)

# 4) Part-of-Speech Tagging

nltk.download('averaged_perceptron_tagger')

def tag_parts_of_speech(text):
    tokens = word_tokenize(text)
    return nltk.pos_tag(tokens)

#Tag parts of speech in the original text
pos_tags = tag_parts_of_speech(your_text)

print("Part-of-Speech Tags:", pos_tags)

# 5) data collection and cleaning 
import re
import string

def preprocess_text(text):
    #remove articles 
    pattern = r'\b(?:a|an|the)\s+'
    text= re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove whitespace
    text = text.strip()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

preprocessed_text = preprocess_text(your_text)
print(preprocessed_text)

# 6)categorization the data (see the precision, accuracy, f1-score and recall)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Hypothetical dataset with an equal number of texts per category
data = [
    # Hydrology
    ("The river basin had a complex network of tributaries", "Hydrology"),
    ("Groundwater levels fluctuate seasonally depending on precipitation", "Hydrology"),
    ("Studies indicate that aquifer depletion has accelerated in recent years", "Hydrology"),
    ("Floodplain mapping is essential for understanding river dynamics", "Hydrology"),
    
    # Environmental Science
    ("Sedimentary rocks form the aquifer's primary structure", "Environmental Science"),
    ("Land use in the region has affected the natural habitats", "Environmental Science"),
    ("Ecosystem services are disrupted by unsustainable water withdrawals", "Environmental Science"),
    ("Biodiversity in wetland areas provides clues about water quality", "Environmental Science"),
    
    # GIS and Remote Sensing
    ("We used GIS to map the flood zones in the area", "GIS and Remote Sensing"),
    ("Remote sensing data helped in assessing the extent of deforestation", "GIS and Remote Sensing"),
    ("GIS technology was crucial in identifying potential sites for groundwater extraction", "GIS and Remote Sensing"),
    ("Satellite imagery analysis reveals changes in land use patterns over time", "GIS and Remote Sensing"),
    
    # Water Resource Management
    ("Policies for water conservation were implemented", "Water Resource Management"),
    ("The impact of climate change on freshwater systems requires adaptive management strategies", "Water Resource Management"),
    ("Water resource managers are developing drought contingency plans", "Water Resource Management"),
    ("Sustainable water management is key to balancing urban and agricultural demands", "Water Resource Management"),
]

#Separate the data into the texts and the labels
texts, labels = zip(*data)

#Splitting the dataset into training and test sets
text_train, text_test, label_train, label_test = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)

#Create the pipeline with a TF-IDF vectorizer and a Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

pipeline.fit(text_train, label_train)

label_pred = pipeline.predict(text_test)

print(classification_report(label_test, label_pred))

# 7) Here we can see how the data is categorized, and by the input we can see its class. 
#Function to classify text using the trained model
def classify_text(new_text):
    predicted_category = model.predict([new_text])[0]
    return predicted_category

user_input = input("Enter a phrase to classify: ")
classification = classify_text(user_input)
print(f"The phrase '{user_input}' is classified as: {classification}")







