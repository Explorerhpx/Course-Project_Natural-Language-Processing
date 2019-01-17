# Extract features from one post & Transform a post into a vector
import numpy as np
import jieba
import re
from snownlp import SnowNLP
import pickle
import codecs

alpha = 0.00001 # smoothing parameter

# Meaning of these function could be found in the definition
# Extract features from origin text
Function_origin = ['URL(post[\'text\'])','First_person(post[\'text\'])',
                   'Pos_micro(post_sentiment)','Neg_micro(post_sentiment)']
# Extract features from preprocessed text
Function = ['Len(post[\'text\'])',# Content-based features
            'Positive_words(post[\'text\'])','Negative_words(post[\'text\'])',
            'Sentiment_score(post[\'text\'],emoticons)',
            'Smiling(emoticons) ','Frowning(emoticons)',
            'Hashtags(post[\'text\'])','AT(post[\'text\'])',
            'Q_mark(post[\'text\'])','E_mark(post[\'text\'])',
            'Multi_QE_mark(post[\'text\'])',
            'Person_description(post)', # User-based features
            'Profile(post)','Verified(post)',
            'Verified_type(post)','Friends(post)',
            'Male(post)','City(post)',
            'Follower(post)','Post_count(post)',
            'Exect_days(post)','Reputation_score(post)',
            'Re_post (post)', # Diffusion-based features
            'Comment_count(post)'
            ]
Dimension = len(Function) + len(Function_origin)

def feature_extract(Post):
    post = Post.copy()
    post_feature = np.zeros([1, Dimension])

    emoticons = {':(':0,':)':0} # smiling (frowning) emoticons
    emoticons[':('] = len(re.findall(':\(',post['text'])) + len(re.findall(':\（',post['text']))+ \
                      len(re.findall('：\(', post['text'])) + len(re.findall('：\（', post['text']))
    emoticons[':)'] = len(re.findall(':\)', post['text'])) + len(re.findall(':\）', post['text'])) + \
                      len(re.findall('：\)', post['text'])) + len(re.findall('：\）', post['text']))

    post_sentiment = 0.5 # Sentiment of microblog
    if (post['text'] != ''):
        post_sentiment = SnowNLP(post['text']).sentiments

    # Extract features before processed text
    for i in range(len(Function_origin)):
        post_feature[0,i] = eval(Function_origin[i])
    # Text processing
    text = {}
    for word in jieba.cut(post['text']):
        if(word in text):
            text[word] += 1
        else:
            text[word] = 1
    post['text'] = text
    for i in range(len(Function)):
        post_feature[0,i+len(Function_origin)] = eval(Function[i])
    return post_feature


## Create sentiment lexicon
# pos_dic = {}
# neg_dic = {}
# with codecs.open('BosonNLP_sentiment_score.txt','r','utf-8') as file:
#     for ite in file.readlines():
#         word,value = ite.split()
#         value = float(value)
#         if (float(value) >= 0):
#             pos_dic[word] = value
#         else:
#             neg_dic[word] = value

## store sentiment lexicon
# Postive_file = open('Positive_dictionary.pkl','wb')
# pickle.dump(pos_dic,Postive_file)
# Negative_file = open('Negative_dictionary.pkl','wb')
# pickle.dump(neg_dic,Negative_file)

# load sentiment lexicon
Pos = open('Positive_dictionary.pkl','rb')
Neg = open('Negative_dictionary.pkl','rb')
positive_dictionary = pickle.load(Pos)
negative_dictionary = pickle.load(Neg)

# Text based features
def URL(text): # Is there any URL in text
    return (re.search('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?='
                      '~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]',text) != None) + alpha

def First_person(text): # First-person pronouns or not
    return ('我'in text[:3] or '我们' in text[:3] or '咱' in text[:3]
            or '咱们' in text[:3] or '本人' in text[:3])

def Pos_micro(post_sentiment): # Positive microblog or not
    return (post_sentiment > 0.69) + alpha

def Neg_micro(post_sentiment): # Negative microblog or not
    return (post_sentiment < 0.31) + alpha

def Positive_words(text): # number of positive words
    account = 0
    for word in text:
        account += word in positive_dictionary
    return account + alpha

def Negative_words(text): # number of negative words
    account = 0
    for word in text:
        account += word in negative_dictionary
    return account + alpha

def Len(text): # length of test
    return (len(text)+alpha)

def Sentiment_score(text,emoticons):
    return Positive_words(text) + emoticons[':)'] - \
           Negative_words(text) - emoticons[':('] + alpha

def Smiling(emoticons): # with smiling emoticons or not
    return (emoticons[':)'] != 0) + alpha

def Frowning(emoticons): # with frowning emoticons or not
    return (emoticons[':('] != 0) + alpha

def Hashtags(text): # # or not
    return '#' in text

def AT(text): # @ or not
    return ('@' in text)+alpha

def Q_mark(text): # ？or not
    return (('？' in text) or ('?'in text ))+ alpha

def E_mark(text): # ! or not
    return (('!' in text) or ('!' in text)) + alpha

def Multi_QE_mark(text): # multiple！？ or not
    Mul_Q = ((('？' in text) and (text['？']>1)) or (('?' in text) and (text['?']>1)))
    Mul_E = ((('!' in text) and (text['!']>1)) or (('！' in text) and (text['！']>1)))
    return (Mul_E or Mul_Q) + alpha

# User_based Feature
def Person_description(post):
    return ( post['user_description'] != "" ) + alpha

def Profile(post): # Profile or not
    return (post['user_avatar'][-1] == '1') + alpha

def Friends(post): # number of followees
    return post["friends_count"] + alpha

def Follower(post): # number of followers
    return post["followers_count"] + alpha

def Exect_days(post): # time length since creating data
    return (post['t'] - post['user_created_at']) / (3600*24) + alpha

def Verified(post): # varied or not
    return post['verified'] + alpha

def Verified_type(post): # varied type
    return (post['verified_type'] == -1) + alpha

def Male(post): # male or not
    return (post['gender'] == 'm')

def City(post): # size of city
    return (post['city'] == '1') + alpha

def Reputation_score(post): # reputation score: followers/followees
    return post['followers_count']/(post['friends_count'] +
                                    (post['friends_count'] == 0))+ alpha

def Post_count(post): # number of posts the user have post
    return post['statuses_count'] + alpha

# Diffusion_based Feature
def Comment_count(post): # number of comment
    return post['comments_count'] + alpha

def Re_post (post): # number of retweets
    return post['reposts_count'] + alpha

def Microblog_count(): # number of microblogs in this period
    return 0

if __name__ == '__main__':
    pass