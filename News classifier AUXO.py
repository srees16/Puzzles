#News classifier AUXO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import initializers, regularizers, constraints
from keras.layers import Dropout, Embedding, MaxPooling1D, Flatten, Concatenate, sequence, Dense, Input, Conv1D
from keras.models import Sequential, Model
from keras.initializers import Constant
from keras.layers.merge import add
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#importing the dataset to read json
dataset=pd.read_json('News_Category_Dataset.json',lines=True)
dataset.head(3)
categories=dataset.groupby('category')
print("No of categories:",categories.ngroups) #grouping the news as per their categories
print(categories.size()) #No of news in each category
dataset.category=dataset.category.map(lambda x:"WORLDPOST" if x=="THE WORLDPOST" else x) #Merging identical categories using lambda (anonymous) function
#using headlines and short_description as input
dataset['text']=dataset.headline+ " "+dataset.short_description
#tokenizing the text in 'text' column
token=Tokenizer()
token.fit_on_texts(dataset.text)
X=token.texts_to_sequences(dataset.text)
dataset['words']=X
#removing empty and short data
dataset['word_length']=dataset.words.apply(lambda i:len(i))
dataset=dataset[dataset.word_length>=5] #removing words less than 5 characters
dataset.head(3)
dataset.word_length.describe() #Word count
#using 50 for padding length
maxlen=50
X=list(sequence.pad_sequences(dataset.words,maxlen=maxlen))
categories=dataset.groupby('category').size().index.tolist() #list of the categories and the size
category_int={}
int_category={}
for i,k in enumerate(categories):
    category_int.update({k:i})
    int_category.update({i:k})
dataset['c2id']=dataset['category'].apply(lambda x:category_int[x])
'''Glove embedding'''
word_index=token.word_index
EMBEDDING_DIM=100
embeddings_index={}
glove_file=open('glove.6B.100d.txt')
for line in glove_file:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
glove_file.close()
print('Found %s unique tokens.' % len(word_index))
print('Total %s word vectors.' % len(embeddings_index))
#Embedding matrix
embedding_matrix=np.zeros((len(word_index)+1,EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector
embedding_layer=Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=Constant(embedding_matrix),input_length=maxlen,trainable=False)
#Dependent and independent variables
X=np.array(X)
Y=np_utils.to_categorical(list(dataset.c2id))
#split to training set and validation set
x_train,x_val,y_train,y_val=train_test_split(X,Y,test_size=0.2,random_state=15)
'''Text CNN'''
inp=Input(shape=(maxlen,),dtype='int32')
embedding=embedding_layer(inp)
stacks=[]
for kernel_size in [2,3,4]:
    conv=Conv1D(64,kernel_size,padding='same',activation='relu',strides=1)(embedding)
    pool=MaxPooling1D(pool_size=3)(conv)
    drop=Dropout(0.5)(pool)
    stacks.append(drop)
merged=Concatenate()(stacks)
flatten=Flatten()(merged)
drop=Dropout(0.5)(flatten)
outp=Dense(len(int_category), activation='softmax')(drop)
NewsNN=Model(inputs=inp, outputs=outp)
NewsNN.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
NewsNN.summary()
textcnn_history=NewsNN.fit(x_train,y_train,batch_size=128,epochs=20,validation_data=(x_val,y_val))
#Accuracy and visualization
acc=textcnn_history.history['acc']
val_acc=textcnn_history.history['val_acc']
loss=textcnn_history.history['loss']
val_loss=textcnn_history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.title('Training and validation accuracy')
plt.plot(epochs,acc,'red',label='Training acc')
plt.plot(epochs,val_acc,'blue',label='Validation acc')
plt.legend()
plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs,loss,'red',label='Training loss')
plt.plot(epochs,val_loss,'blue',label='Validation loss')
plt.legend()
plt.show()
#confusion matrix
predicted=NewsNN.predict(x_val)
cm=pd.DataFrame(confusion_matrix(y_val.argmax(axis=1),predicted.argmax(axis=1)))
from IPython.display import display
pd.options.display.max_columns = None
display(cm)
#evaluate accuracy
def evaluate_accuracy(model):
    predicted = model.predict(x_val)
    diff = y_val.argmax(axis=-1) - predicted.argmax(axis=-1)
    corrects = np.where(diff == 0)[0].shape[0]
    total = y_val.shape[0]
    return float(corrects/total)
print("model NewsNN accuracy: %.6f" % evaluate_accuracy(NewsNN))
#Evaluate accuracy
def evaluate_accuracy_ensemble(models):
    res=np.zeros(shape=y_val.shape)
    for model in models:
        predicted=model.predict(x_val)
        res+=predicted
    res/=len(models)
    diff=y_val.argmax(axis=-1) - res.argmax(axis=-1)
    corrects=np.where(diff == 0)[0].shape[0]
    total=y_val.shape[0]
    return float(corrects/total)
print(evaluate_accuracy_ensemble(NewsNN))