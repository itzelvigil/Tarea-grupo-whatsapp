import pandas as pd
import re
import emoji
import numpy as np
import wordcloud 
import nltk
import matplotlib.pyplot as plt
import plotly.express as px
import collections
import regex
from collections import Counter

def startsWithDateAndTime(s):
    # regex pattern for date.(Works only for android. IOS Whatsapp export format is different. Will update the code soon
    pattern = r'^\d{1,2}/\d{1,2}/\d{2} \d{1,2}:\d{2}'
    result = re.match(pattern, s)
    if result:
        return True
    return False
  
# Finds username of any given format.
def FindAuthor(s):
    patterns = [
        '([\w]+):',                        # First Name
        '([\w]+[\s]+[\w]+):',              # First Name + Last Name
        '([\w]+[\s]+[\w]+[\s]+[\w]+):',    # First Name + Middle Name + Last Name
        '([+]\d{2} \d{5} \d{5}):',         # Mobile Number (India)
        '([+]\d{2} \d{3} \d{3} \d{4}):',   # Mobile Number (US)
        '([\w]+)[\u263a-\U0001f999]+:',
        '([\w]+) [^\s\u1f300-\u1f5ff]*'    # Name and Emoji              
    ]
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, s)
    if result:
        return True
    return False
  
def getDataPoint(line):   
    splitLine = line.split(' - ') 
    dateTime = splitLine[0]
    date, time, aux = dateTime.split(' ') 
    message = ' '.join(splitLine[1:])
    if FindAuthor(message): 
        splitMessage = message.split(': ') 
        author = splitMessage[0] 
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    return date, time, author, message

parsedData = [] # List to keep track of data so it can be used by a Pandas dataframe
# Upload your file here
conversationPath = 'SinGraduacionISC.txt' # chat file
with open(conversationPath, encoding="utf-8") as fp:    
    fp.readline() # Skipping first line of the file because contains information related to something about end-to-end encryption
    messageBuffer = [] 
    date, time, author = None, None, None
    while True:
        line = fp.readline() 
        if not line: 
            break
        line = line.strip() 
        if startsWithDateAndTime(line): 
            if len(messageBuffer) > 0: 
                parsedData.append([date, time, author, ' '.join(messageBuffer)]) 
            messageBuffer.clear() 
            date, time, author, message = getDataPoint(line) 
            messageBuffer.append(message) 
        else:
            messageBuffer.append(line)
   
df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message']) # Initialising a pandas Dataframe.
df["Date"] = pd.to_datetime(df["Date"])
df['Message'] = df['Message'].str.lower() # Mensajes en minúsculas
df['Message'] = df.Message.str.replace(r"(a|j)?(ja)+(a|j)?", "jaja") 
df['Message'] = df.Message.str.replace(r"(a|h)?(ha)+(a|h)?", "jaja") 
df = df.dropna()

# reemplaza los nombres de los usuarios con nombres de la serie Avatar: La leyenda de Aang
avatar_dt = pd.read_csv("Avatar2.csv") 

# Obtiene nombres y aliases
nombres = list(df.Author.unique())
aliases = list(avatar_dt.name.sample(len(nombres)))
print(aliases)

df.Author.replace(nombres, aliases, inplace=True)

users = df.Author.unique()
print(users)


def split_count(text):

    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list

total_messages = df.shape[0]
media_messages = df[df['Message'] == '<multimedia omitido>'].shape[0]
df["emoji"] = df["Message"].apply(split_count)
emojis = sum(df['emoji'].str.len())
URLPATTERN = r'(https?://\S+)'
df['urlcount'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
links = np.sum(df.urlcount)

print("Group wise stats")
print("Mensajes: ", total_messages)
print("Media: ", media_messages)
print("Emojis: ", emojis)
print("Links: ",links)

media_messages_df = df[df['Message'] == '<multimedia omitido>']
messages_df = df.drop(media_messages_df.index)
messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
messages_df["MessageCount"]=1

print("******************************** \n")

# Creates a list of unique Authors - ['Manikanta', 'Teja Kura', .........]
l = messages_df.Author.unique()

for i in range(len(l)):
  # Filtering out messages of particular user
  req_df= messages_df[messages_df["Author"] == l[i]]
  # req_df will contain messages of only one particular user
  print(f'Estadisticas de: {l[i]} -')
  # shape will print number of rows which indirectly means the number of messages
  print('Mensajes enviados', req_df.shape[0])
  #Word_Count contains of total words in one message. Sum of all words/ Total Messages will yield words per message
  words_per_message = (np.sum(req_df['Word_Count']))/req_df.shape[0]
  print('Palabras por mensaje ', words_per_message)
  #media conists of media messages
  media = media_messages_df[media_messages_df['Author'] == l[i]].shape[0]
  print('Mensajes con archivos enviados ', media)
  # emojis conists of total emojis
  emojis = sum(req_df['emoji'].str.len())
  print('Emojis enviados', emojis)
  #links consist of total links
  links = sum(req_df["urlcount"])   
  print('Links enviados ', links)   
  print()

total_emojis_list = list(set([a for b in messages_df.emoji for a in b]))
total_emojis = len(total_emojis_list)
print(total_emojis)

total_emojis_list = list([a for b in messages_df.emoji for a in b])
emoji_dict = dict(Counter(total_emojis_list))
emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)

emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
print(emoji_df)



text = " ".join(review for review in messages_df.Message)
print ("Hay {} palabras en todos los mensajes.".format(len(text)))

from nltk.corpus import stopwords
nltk.download("stopwords")
from wordcloud import WordCloud 
stopwords = set(stopwords.words('spanish', 'english')) 
stopwords.update([ "jaja", "com", "http","https", "www"]) #además de las palabras en el diccionario podemos agregar otras según sea necesario
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

from PIL import Image
import urllib
import requests
def plot_cloud(wordcloud): 
    # Definimos parámetros del gráfico
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off")
    plt.show();

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plot_cloud(wordcloud)


date_df = messages_df.groupby("Date").sum()
date_df.reset_index(inplace=True)
print(date_df)
fig = px.line(date_df, x="Date", y="MessageCount", labels={'Date':'Fecha', 'MessageCount':'Mensajes'}, title='Número de mensajes a través del tiempo.')
fig.update_xaxes(nticks=20)
import plotly.offline as py
py.plot(fig, filename = 'linea_de_tiempo.html', auto_open=True)

fig1 = px.pie(emoji_df, values='count', names='emoji',
            title='Distribución de Emoji')
fig1.update_traces(textposition='inside', textinfo='percent+label')
py.plot(fig1, filename = 'emogis.html', auto_open=True)


l = messages_df.Author.unique()
for i in range(len(l)):
  dummy_df = messages_df[messages_df['Author'] == l[i]]
  total_emojis_list = list([a for b in dummy_df.emoji for a in b])
  emoji_dict = dict(Counter(total_emojis_list))
  emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
  print('Emoji Distribution for', l[i])
  author_emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
  fig2 = px.pie(author_emoji_df, values='count', names='emoji')
  fig2.update_traces(textposition='inside', textinfo='percent+label')
  py.plot(fig1, filename = 'emogis2.html', auto_open=True)



messages_df['Date'].value_counts().head(10).plot.barh()
plt.xlabel('Number of Messages')
plt.ylabel('Date')
plt.show()


def dayofweek(i):
  l = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  return l[i];
day_df=pd.DataFrame(messages_df["Message"])
day_df['day_of_date'] = messages_df['Date'].dt.weekday
day_df['day_of_date'] = day_df["day_of_date"].apply(dayofweek)
day_df["messagecount"] = 1
day = day_df.groupby("day_of_date").sum()
day.reset_index(inplace=True)

fig3 = px.line_polar(day, r='messagecount', theta='day_of_date', line_close=True)
fig3.update_traces(fill='toself')
fig3.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0,300]
    )),
  showlegend=False
)
py.plot(fig3, filename = 'emogis2.html', auto_open=True)