from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
class ChatBot:
    def __init__(self):
        self.labels = []
        self.questions = []
        self.answers = []
        self.bow_vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
    def format_domain(self):
        for line in open('que.txt',encoding = "utf8"):
            self.labels.append(line.strip().split(" ")[-1])
            self.questions.append(" ".join(line.strip().split(" ")[:-1]))
        for line in open('ans.txt', encoding = "utf8"):
            self.answers.append(line.strip())
    def train_bot(self):
        training_vectors = self.bow_vectorizer.fit_transform(self.questions)
        self.classifier.fit(training_vectors, self.labels)
    def fetch_answer(self,query):
        input_vector = self.bow_vectorizer.transform([query])
        predict = self.classifier.predict(input_vector)
        index = int(predict[0])
        accuracy = str(self.classifier.predict_proba(input_vector)[0][index-1] * 100)[:5] + "%"
        return self.answers[index-1],accuracy
bot = ChatBot()
bot.format_domain()
bot.train_bot()
while(True):
    command = input("User:# ")
    answer,accuracy = bot.fetch_answer(command)
    print("rBot:# ",answer, "    [",accuracy,"]")
