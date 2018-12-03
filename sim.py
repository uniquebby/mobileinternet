#文本相似度计算
#使用TF-IDF计算文本相似度

from gensim import corpora,models,similarities

import jieba
from collections import defaultdict

f=open("词频.txt","w")
stopkey=[line.strip() for line in open('stopword.txt').readlines()]#加载停用词
doc1="doc1.txt"
doc2="doc2.txt"
doc3="doc3.txt"
doc4="doc4.txt"
doc5="doc5.txt"
doc6="doc6.txt"
d1=open(doc1).read()
d2=open(doc2).read()
d3=open(doc3).read()
d4=open(doc4).read()
d5=open(doc5).read()
d6=open(doc6).read()
#分词
data1=jieba.cut(d1.strip())
data2=jieba.cut(d2.strip())
data3=jieba.cut(d3.strip())
data4=jieba.cut(d4.strip())
data5=jieba.cut(d5.strip())
data6=jieba.cut(d6.strip())

data11=""
for item in data1:
    if item not in stopkey:
        data11+=item+" "
data21=""
for item in data2:
    if item not in stopkey:
        data21+=item+" "
data31=""
for item in data3:
    if item not in stopkey:
        data31+=item+" "
data41=""
for item in data4:
    if item not in stopkey:
        data41+=item+" "
data51=""
for item in data5:
    if item not in stopkey:
        data51+=item+" "
data61=""
for item in data6:
    if item not in stopkey:
        data61+=item+" "

documents=[data11,data21,data31,data41,data51,data61]
texts=[[word for word in document.split()]
       for document in documents]
#统计各个词的频率
frequency=defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1
#将词频保存文档中
f.write("词项 词频\n")
for item in frequency.keys():
    f.write(item+"  "+str(frequency[item])+"\n")
#构造词典
dictionary=corpora.Dictionary(texts)

new_vec1=dictionary.doc2bow(data11.split())#在词典中出现位置和频次
new_vec2=dictionary.doc2bow(data21.split())
new_vec3=dictionary.doc2bow(data31.split())
new_vec4=dictionary.doc2bow(data41.split())
new_vec5=dictionary.doc2bow(data51.split())
new_vec6=dictionary.doc2bow(data61.split())

corpus=[dictionary.doc2bow(text) for text in texts]
tfidf=models.TfidfModel(corpus)#文档数，词典词项数+1
new_tfidf = tfidf[corpus]
featureNum=len(dictionary.token2id.keys())#词典中词项数目
index=similarities.SparseMatrixSimilarity(new_tfidf,num_features=featureNum)

new_vec_tfidf1 = tfidf[new_vec1]
new_vec_tfidf2 = tfidf[new_vec2]
new_vec_tfidf3 = tfidf[new_vec3]
new_vec_tfidf4 = tfidf[new_vec4]
new_vec_tfidf5 = tfidf[new_vec5]
new_vec_tfidf6 = tfidf[new_vec6]
#print(new_vec_tfidf)
sim1=index[new_vec_tfidf1]
sim2=index[new_vec_tfidf2]
sim3=index[new_vec_tfidf3]
sim4=index[new_vec_tfidf4]
sim5=index[new_vec_tfidf5]
sim6=index[new_vec_tfidf6]
print("文档1对应相关性：")
print(sim1)
print("文档2对应相关性：")
print(sim2)
print("文档3对应相关性：")
print(sim3)
print("文档4对应相关性：")
print(sim4)
print("文档5对应相关性：")
print(sim5)
print("文档6对应相关性：")
print(sim6)
