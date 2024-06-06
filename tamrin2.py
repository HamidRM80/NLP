import nltk

nltk.download("punkt")

paragraph = """In the small coastal town of Havenport, where the salty breeze whispered secrets of the sea, lived a young girl named Elara. With her fiery red hair and inquisitive green eyes, she was known to the townsfolk as the curious dreamer. Elara's life revolved around the stories her grandmother, Nona, told her each night by the fireplace.
One particularly stormy evening, as rain lashed against the windows and thunder rumbled in the distance, Nona handed Elara a small, intricately carved silver locket. "This locket," Nona began, her voice trembling slightly, "has been in our family for generations. Inside it lies a secret that only the bravest of our kin have discovered."
Elara's eyes widened with wonder as she carefully opened the locket, revealing a tiny, delicate map. The parchment was old and faded, yet the intricate details were still visible. "What is this map, Nona?" she asked, her voice filled with excitement.
"It's a map to the legendary Isle of Liora," Nona explained. "A place said to hold treasures beyond imagination, but also perils that test the heart and spirit."
The next morning, Elara woke up with a sense of purpose. Clutching the locket tightly, she set off towards the harbor where she met her best friend, Finn, a spirited boy with a knack for adventure. When Elara shared the story of the locket, Finn's eyes sparkled with enthusiasm. "We have to find this island!" he exclaimed.
Together, they gathered supplies and set sail on Finn's small but sturdy boat, the Sea Breeze. The journey was arduous, with choppy waves and unpredictable weather. Yet, the thought of the treasure and the spirit of adventure kept their hearts warm.
After days at sea, guided by the ancient map and the constellations, they finally caught sight of the mist-shrouded Isle of Liora. As they approached the shore, they were met with an eerie silence, broken only by the occasional cry of a seagull.
The island was a lush paradise, teeming with vibrant flora and fauna. Elara and Finn followed the map, which led them through dense forests and over rocky hills. Along the way, they encountered challenges that tested their courage and friendship. They crossed rickety bridges, solved ancient riddles, and navigated treacherous paths.
At last, they reached a hidden cave, its entrance adorned with glowing crystals. Inside, they found an enormous chest, covered in dust and cobwebs. With trembling hands, Elara opened the chest, revealing a dazzling array of jewels, gold coins, and ancient artifacts.
But among the riches, what caught Elara's eye was a simple, worn journal. As she opened it, she realized it was written by one of her ancestors, chronicling their journey and the lessons they learned. The final entry read, "The greatest treasure is not the gold or jewels, but the wisdom and courage gained through our journey."
Elara and Finn smiled at each other, understanding the true meaning of their adventure. They took a few jewels as mementos and decided to leave the rest, ensuring that future explorers would have a chance to discover the island's secrets.
Returning to Havenport, Elara felt a sense of fulfillment. She knew the stories she would share now had a deeper meaning, enriched by her own experiences. And as for the silver locket, it remained a cherished heirloom, a reminder of the adventure that taught her the greatest treasure lay within her own heart."""

sentences = nltk.sent_tokenize(paragraph)

# print(paragraph)

words = nltk.word_tokenize(paragraph)

# print(words)

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

stemmer = PorterStemmer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)

# print(sentences[1])

from nltk.stem import WordNetLemmatizer
nltk.download('wordnetcls')

lemmatizer = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)

# print(sentences[1])

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()

# print(x)

sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x = tf.fit_transform(corpus).toarray()
