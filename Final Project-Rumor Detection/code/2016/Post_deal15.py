# Extract features from one post & Transform a post into a vector
import numpy as np
import jieba
import re
import pickle
import codecs

alpha = 0.0001 # smoothing parameter

# Meaning of these function could be found in the definition
# Extract features from preprocessed text
Function = ['Len(post[\'text\'])',# Content-based features
            'Hashtags(post[\'text\'])','AT(post[\'text\'])',
            'Q_mark(post[\'text\'])','E_mark(post[\'text\'])',
            'Multi_QE_mark(post[\'text\'])',
            'Person_description(post)', # User-based features
            'Verified(post)',
            'Verified_type(post)','Friends(post)',
            'Male(post)','City(post)',
            'Follower(post)','Post_count(post)',
            'Exect_days(post)',
            'Re_post (post)', # Diffusion-based features
            'Comment_count(post)']

Dimension = len(Function)

def feature_extract(Post):
    post = Post.copy()
    post_feature = np.zeros([1, Dimension])

    # Text processing
    text = {}
    for word in jieba.cut(post['text']):
        if(word in text):
            text[word] += 1
        else:
            text[word] = 1
    post['text'] = text
    for i in range(len(Function)):
        post_feature[0,i] = eval(Function[i])
    return post_feature

# Text based features
def Len(text): # length of test
    return (len(text)+alpha)

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

def Friends(post): # 朋友数
    return post["friends_count"] + alpha

def Follower(post): # 被follow 数
    return post["followers_count"] + alpha

def Exect_days(post): # 创建后存在天数
    return (post['t'] - post['user_created_at']) / (3600*24) + alpha

def Verified(post): # 是否验证身份
    return post['verified'] + alpha

def Verified_type(post): # 身份类型(是否为普通用户)
    return (post['verified_type'] == -1) + alpha

def Male(post): #是否是男性
    return (post['gender'] == 'm')

def City(post): #所在城市大小
    return (post['city'] == '1') + alpha

def Post_count(post): # 用户发送微博数
    return post['statuses_count'] + alpha

# Diffusion_based Feature
def Comment_count(post): # number of comment
    return post['comments_count'] + alpha

def Re_post (post): # number of retweets
    return post['reposts_count'] + alpha

if __name__ == '__main__':
    pass