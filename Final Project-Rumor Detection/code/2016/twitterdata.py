import tweepy
import json
import sys
import re
import chardet
# export http_proxy='http://localhost:8118'
# export https_proxy='http://localhost:8118'

def progress_test(index, length):
    bar_length=20
    hashes = '#' * int(index/length * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [%s] %d/%d"%(hashes + spaces, index, length))
    sys.stdout.flush()

consumer_key = "fgb3qh9esWPRGLWeQzHI5c3WK"
consumer_secret = "Bhs2fYvkTZlMh2dUKdxNpDfXSU3P2JCk7F5KkgOYMSjhLcya9J"
access_token = "878097078731161600-n9bGoyiVOxt5b393HheHQshkT71nWHi"
access_token_secret = "cA4V92saUdIIY5ySpDTiTLFl9Z8n4XxjbxARGzsGrNU2X"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
# id_list = [677641866930757635, 677644704100544513, 677653639389962241]
# result = api.statuses_lookup(id_list)
# print(result[0].created_at)
file = open('../trainSet_twitter.txt', 'r')
l = file.readlines()
file.close()
file = open('../Twitter.txt','r')
ll = file.readlines()
file.close()
Events = dict()
for each in ll:
	each.strip('\n')
	s = each.split('\t')
	eve = re.sub('eid:','',s[0]).strip('\n')
	label = re.sub('label:','',s[1]).strip('\n')
	idstr = s[2].strip('\n')
	tmp = dict()
	tmp['label'] = label
	tmp['idstr'] = idstr
	Events[eve] = tmp
index = 0
for each in l:
	index += 1
	progress_test(index, len(l))
	each.strip('\n')
	filepath = each
	each  = re.sub('.json', '', each).strip('\n')
	Events[each]['idstr'] = idstr
	idstr = idstr.rstrip()
	id_list = [ int(i) for i in idstr.split(' ')]
	result = api.statuses_lookup(id_list)
	search_result =[{'text':status.text, 'created_at':str(status.created_at)} for status in result]
	with open(filepath.strip('\n'),'w') as f:
		sys.stdout = f
		for res in search_result:
			print(res)
		sys.stdout = sys.__stdout__

# if __name__ == '__main__':
# 	f = open('Airfrance.json','r')
# 	fencode = chardet.detect(f.read())
# 	print(fencode)
