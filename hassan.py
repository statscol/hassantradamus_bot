import numpy as np
import nltk 
import string
import unidecode
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import (Bot, ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler,ConversationHandler)
import json

###NLKT COMPONENTS NEEDED
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords

import tweepy
auth = tweepy.OAuthHandler(os.environ['TW_CONS_KEY'], os.environ['TW_CONS_SEC'])
auth.set_access_token(os.environ['TW_ATOKEN'], os.environ['TW_ATOKEN_SEC'])
api = tweepy.API(auth)

f=open("hassantweets.txt",encoding="utf'8",mode="r")
#f=open("c:/users/jhonp/desktop/hassanfiles/hassantweets.txt",encoding="utf'8",mode="r")
datos=[]
for line in f:
    ##Se quitan los acentos previo a la concatenación de
    datos.append(unidecode.unidecode(line.replace('"','').replace("\n","")).split("\t"))
   
##skip first line with headers
datos=np.array(datos[1:]).reshape((-1,7))  

##clean text
def clean_text(text_aux):
    table = str.maketrans('', '', string.punctuation)
    stripped = text_aux.translate(table).lower().replace("  ","").translate(str.maketrans('','', '0123456789')).strip()
    return(stripped)
    
texto=[clean_text(i) for i in datos[:,3]]

datos=np.append(datos,np.reshape(texto,(-1,1)),axis=1)
datos=datos[datos[:,7]!=""]

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def most_related_to(user_response,text_corpus=datos[:,7]):
    text_aux=" . ".join(text_corpus) 
    st = nltk.sent_tokenize(text_aux)# converts to list of sentences 
    response=''
    st.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords.words("spanish"))
    tfidf = TfidfVec.fit_transform(st)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-6:]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    out_text=[i for i,text in enumerate(st) if i in idx][:-1]
    idt=datos[np.random.choice(out_text,1)[0],1]
    tweet=api.get_status(idt)
    frase,fecha=tweet.text,tweet.created_at
    link="https://twitter.com/HassNassar/status/"+idt
    if req_tfidf==0:
        return(np.nan)
    else:
        return(f"Esto dije el {str(fecha.date())}: {frase} ,enlace: {link} ")

def trends(*args):
    if len(args)==0:
        trends=api.trends_place("368148") ##BOGOTA,COLOMBIA WOEID
        trends=json.loads(json.dumps(trends, indent=1))
        names=[]
        for i in trends[0]['trends']:
            names.append(unidecode.unidecode(clean_text(i['name'])))
        #selecciona aleatoriamente una de las 6 tendencias actuales en twitter.
        sel_trend=np.random.choice(names[1:8],1)[0]
        res=most_related_to(sel_trend)
        if res is np.nan:
            return(f"No tengo nada que decir acerca de {sel_trend}")
        else:
            return(f"{sel_trend} es tendencia, {res} ")
    else:
        res=most_related_to(str(args[0]))
        if res is np.nan:
            return(f"No tengo nada que decir acerca de {str(args[0])}")
        else:
            return(f"Respecto a {str(args[0])}, considero: {res} ")


#### BOT FUNCTIONS FOR INTERACTION

def response_text(bot,update):
    user_response=unidecode.unidecode(clean_text(update.message.text)) ##clean text before moving it to the corpus
    update.message.reply_text(most_related_to(user_response),reply_markup=ReplyKeyboardRemove())

def voicemsg(bot, update):
    update.message.reply_text("Aún no puedo responder tu audio con mi sabiduría",reply_markup=ReplyKeyboardRemove())

def hoy(bot, update):
    update.message.reply_text(trends(),reply_markup=ReplyKeyboardRemove())


def start(bot, update):
    chat_id = update.message.chat_id
    first_msg="Hola Soy Hassantradamus, puedo decirte qué va a pasar con el actual gobierno colombiano con mis opiniones del pasado, si pones /hoy te diré mis predicciones de hoy o escríbeme sobre lo que desees saber y te responderé con una predicción sobre el tema"
    bot.send_message(chat_id=chat_id, text=first_msg)

def main():
    updater = Updater(os.environ['TELEGKEY'])
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start',start))
    dp.add_handler(CommandHandler('hoy',hoy))
    ##text response
    
    resp_handler = MessageHandler(Filters.text, response_text)
    dp.add_handler(resp_handler)
    ##audio response
    dp.add_handler(MessageHandler(Filters.voice, voicemsg))
    
    updater.start_polling()
    updater.idle()
    
   
    
if __name__ == '__main__':
    main()