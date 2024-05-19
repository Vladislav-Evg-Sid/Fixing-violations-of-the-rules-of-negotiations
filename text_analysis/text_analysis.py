import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import spacy


class text_analysis:
    def __init__(self):
        self.nlp = spacy.load("ru_core_news_sm")
        self.columns = [
            'верно', 'верный', 'диспетчер', 'днц', 'допустимый', 'дсп', 'ехать', 'запрещающий', 'здравствовать',
            'куча', 'маршрут', 'машинист', 'номер', 'ой', 'остановиться', 'остановка', 'перегон', 'пикет', 'поезд',
            'понятный', 'понять', 'приказ', 'принимать', 'принять', 'пс', 'светофор', 'сигнал', 'слушать', 'слушаю',
            'слушая', 'сможете', 'смочь', 'спасибо', 'станция', 'стараться', 'тчп', 'удобный'
        ]
        self.workers = {"пс":"ПС", "днс":"ДНЦ", "тчп":"ТЧП", "дсп":"ДСП", "диспетчер":"ДНЦ", "машинист":"ТЧП", "дежурный":"ДСП"}
        self.lower_workers = ["пс", "днц", "тчп", "дсп", "диспетчер", "машинист", "дежурный"]
        df = pd.read_csv('text_analysis/bd.csv', sep=';')
        df_vectors_BoT = self.str2vec_BoT(df)
        self.scaler = StandardScaler()
        self.scaler.fit(df_vectors_BoT)
        df_vectors_BoT = self.scaler.transform(df_vectors_BoT)
        self.break_point = 0.7
        self.model = LogisticRegression()
        self.model.fit(df_vectors_BoT, df['score'])

    def str2vec_BoT(self, df):
        dataset = df['text']
        doc = self.nlp(str(dataset.iloc[0]))
        df1 = pd.DataFrame(
            [" ".join([token.lemma_ for token in doc if not token.is_stop and token.pos_ != "PUNCT"])])
        df1.columns = ['pred']
        for i in range(1, len(dataset)):
            doc = self.nlp(str(dataset.iloc[i]))
            df1.loc[i] = [" ".join([token.lemma_ for token in doc if not token.is_stop and token.pos_ != "PUNCT"])]
        vectors = pd.DataFrame(columns=self.columns)
        for i in range(len(df1)):
            ser = []
            for sub_s in self.columns:
                ser.append(df1.iloc[i][0].count(sub_s))
            vectors.loc[len(vectors)] = ser
        return vectors

    def ret_pred(self, y_pred_dop):
        y_pred = []
        for i in y_pred_dop[:, 1]:
            if i < self.break_point:
                y_pred.append(0)
            else:
                y_pred.append(1)
        return y_pred

    def error_processing(self, text):
        errors = []
        speaker = []
        for i in range(len(text)):
            line = text[i].lower()
            if i == 0:
                flag = True
                for work in self.lower_workers:
                    if work in line:
                        if flag:
                            flag = False
                            speaker.append(self.workers[work])
                        else:
                            if len(speaker) == 1:
                                speaker.append(self.workers[work])
                            else:
                                speaker[1] = self.workers[work]
                if flag:
                    text[i] = '<' + text[i] + '>'
                    errors.append('Нет корректного обращения')
            if i == 1:
                line = text[1].lower()
                flag = True
                for word in ("слуша", "принял", "понял"):
                    if word in line:
                        flag = False
                        break
                if flag:
                    text[1] = '<' + text[1] + '>'
                    errors.append('Нет сообщения о прослушивании волны/принятии информации')
            line = text[i].lower().split(' ')
            for trash in ("спасибо", "пожалуйста", "привет", "пока", "до свидания", "здравствуй"):
                if trash in line:
                    text[i] = text[i].split(' ')
                    for l in range(len(line)):
                        if trash in line[l]:
                            text[i][l] = '<' + text[i][l] + '>'
                            errors.append('Обнаружено слово, неупотребимой в регламенте')
                    text[i] = ' '.join(text[i])
        line = text[-1].lower().split(' ')
        flag = True
        for word in ("выплня", "принял", "понял", "понятно", "верно"):
            if word in line:
                flag = False
        if flag:
            text[-1] = '<' + text[-1] + '>'
            errors.append('Нет сообщения о принятии/верности сообщения')
        if len(speaker) != 2:
            speaker = ["", ""]
            for i in range(1, len(text)):
                line = text[i].lower()
                for work in self.lower_workers:
                    if work in line:
                        speaker[(i + 1) % 2] = self.workers[work]
                if not ('' in speaker):
                    break
            if speaker[0] == '':
                speaker[0] = 'Второй'
            if speaker[1] == '':
                speaker[1] = 'Первый'
        for i in range(len(text)):
            text[i] = speaker[(i + 1) % 2] + ': ' + text[i]
        return text, errors

    def processing(self, text):
        speaker = ["", ""]
        for i in range(1, len(text)):
            line = text[i].lower()
            for work in self.lower_workers:
                if work in line:
                    speaker[(i+1)%2] = self.workers[work]
            if not('' in speaker):
                break
        if speaker[0] == '':
            speaker[0] = 'Второй'
        if speaker[1] == '':
            speaker[1] = 'Первый'
        for i in range(len(text)):
            text[i] = speaker[(i+1)%2] + ': ' + text[i]
        return text, []

    def format(self, text, errors):
        otch = '\n'.join(text)
        if errors == []:
            otch = [otch, 0]
        else:
            otch = [otch, 1, errors]
        return otch

    def analyze(self, text):
        vector = ''
        for i in text:
            vector += i + ' '
        df = pd.DataFrame([vector])
        df.columns = ['text']
        vector = self.str2vec_BoT(df)
        y_pred = self.model.predict(vector)
        if y_pred == 0:
            text, errors = self.error_processing(text)
        else:
            text, errors = self.processing(text)
        text = self.format(text, errors)
        return text

