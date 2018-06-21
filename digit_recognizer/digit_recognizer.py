import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import pylab
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder    
# https://www.kaggle.com/c/digit-recognizer

class digitRecognizer:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None,28,28,1])
        self.y = tf.placeholder(tf.float32,[None,10])
    
        self.w1 = tf.Variable(tf.random_normal([5,5,1,32]))
        self.b1 = tf.Variable(tf.random_normal([32]))
    
        self.w2 = tf.Variable(tf.random_normal([5,5,32,64]))
        self.b2 = tf.Variable(tf.random_normal([64]))

        self.w3 = tf.Variable(tf.random_normal([5,5,64,64]))
        self.b3 = tf.Variable(tf.random_normal([64]))

        self.wf_1 = tf.Variable(tf.random_normal([4*4*64,1024]))
        self.bf_1 = tf.Variable(tf.random_normal([1024]))
    
        self.keep_prob = tf.placeholder(tf.float32)
           
        self.wf_2 = tf.Variable(tf.random_normal([1024,10]))
        self.bf_2 = tf.Variable(tf.random_normal([10]))
        self.learn_rate = tf.placeholder(tf.float32) #0.001
        self.initial_learn_rate = 0.001
        self.cur_learn_rate = 0.001
        self.train_batch_size = 100
        self.VALID = 10000
    
    def initDataFromCSV(self,filestr,t=''):
        csvDF = pd.read_csv(filestr)
        if t == 'train':
            self.train_data = csvDF.drop(columns = ['label']).values.reshape(-1,28,28,1).astype('float32')
            self.train_labels = csvDF['label'].values
            self.train_labels = LabelEncoder().fit_transform(self.train_labels)[:, None]
            self.train_labels = OneHotEncoder().fit_transform(self.train_labels).todense()
        if t == 'test':
            self.test_data = csvDF.values.reshape(-1,28,28,1).astype('float32')

    def splitTrainValidation(self,sec):
        self.valid_data = self.train_data[self.VALID*sec:self.VALID*(sec+1)]
        self.train_data = np.delete(self.train_data,np.s_[self.VALID*sec:self.VALID*(sec+1)],0)
                       
        self.valid_labels = self.train_labels[self.VALID*sec:self.VALID*(sec+1)]
        self.train_labels = np.delete(self.train_labels,np.s_[self.VALID*sec:self.VALID*(sec+1)],0)

    def createNetTensors(self):
        co1 = self.conv2d(self.x, self.w1, self.b1, k=2)
        co2 = self.conv2d(co1, self.w2, self.b2, k=2)   #co2.shape = [,7,7,64]
        co3 = self.conv2d(co2, self.w3, self.b3, k=2)
        co3 = tf.reshape(co3,[-1,4*4*64])
        fc = tf.nn.relu(tf.matmul(co3,self.wf_1) + self.bf_1)
        fc_keep = tf.nn.dropout(fc, self.keep_prob)
        logits  = tf.matmul(fc_keep, self.wf_2) + self.bf_2
        self.prediction = tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y))
         
        self.optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)
        #self.optimizer = tf.train.RMSPropOptimizer(self.learn_rate).minimize(self.loss)
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def conv2d(self,X,W,b,k=2):
        conv = tf.nn.conv2d(X,W,strides=[1,1,1,1], padding='SAME')
        conv = tf.nn.bias_add(conv,b)
        conv = tf.nn.relu(conv)
        conv = tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        return conv
    
    def normalizeTrainValidTestData(self):
        self.train_data = self.train_data/255.
        self.valid_data = self.valid_data/255.
        self.test_data = self.test_data/255.
        
    def augTrainData(self, aug_times=1):
        print ("Before augmentaion: {0}, {1}".format(self.train_data.shape,self.train_labels.shape))
        datagen = ImageDataGenerator(rotation_range=10,zoom_range = 0.1,
                                     width_shift_range=0.1,height_shift_range=0.1)
        imgs_flow = datagen.flow(self.train_data.copy(), batch_size=len(self.train_data), shuffle = False)
        labels_aug = self.train_labels.copy()
        for i in range(aug_times):
            imgs_aug = imgs_flow.next()
            self.train_data = np.append(self.train_data,imgs_aug,0)
            self.train_labels = np.append(self.train_labels,labels_aug,0)
        print ("After augmentaion: {0}, {1}".format(self.train_data.shape,self.train_labels.shape))
        try:
            print ("Trying to show augmentation effects")
            fig,axs = plt.subplots(5,10, figsize=(15,9))
            axs[0,1].imshow(self.train_data[len(self.train_data)//1000*1].reshape(28,28), cmap=cm.binary)
            axs[0,2].imshow(self.train_data[len(self.train_data)//1000*500].reshape(28,28), cmap=cm.binary)
            axs[0,3].imshow(self.train_data[len(self.train_data)//1000*999].reshape(28,28), cmap=cm.binary)
            plt.show()
        except:
            print ("No X server, skipping drawing")

    def initSession(self):
        self.sess = tf.Session()
    
    def closeSession(self):
        self.sess.close()    

    def updateLearnRate(self, ndecay):
        self.cur_learn_rate = self.initial_learn_rate * pow(0.9,ndecay//10)

    def train(self, epochs=0):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        '''data_queue, labels_queue = tf.train.batch([self.train_data,self.train_labels],
                                                     batch_size=self.train_batch_size,
                                                     enqueue_many=True)'''
        data_queue, labels_queue = tf.train.shuffle_batch([self.train_data,self.train_labels],
                                                           batch_size=self.train_batch_size,
                                                           capacity=50000,
                                                           min_after_dequeue=10000,
                                                           enqueue_many=True)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess,coord)
        steps_per_epoch = int(self.train_data.shape[0]/self.train_batch_size)
        if not epochs > 0:
            epochs = 2 
        steps = steps_per_epoch * epochs
        print ("All steps:  {0}*{1} = {2}".format(steps_per_epoch,epochs,steps))
        earlyStopProc = False
        noImproveCount = 0
        cur_epoch = 1
        for step in range(1,steps+1):
            batch_x, batch_y = self.sess.run([data_queue, labels_queue])
            self.sess.run(self.optimizer,feed_dict={self.x:batch_x,
                                                    self.y:batch_y, 
                                                    self.learn_rate: self.cur_learn_rate,
                                                    self.keep_prob: 0.75}) 
            if step % 100 == 0 or step == 1:
                #Show current status 
                lossval, accval = self.sess.run([self.loss, self.accuracy], feed_dict={self.x: batch_x, 
                                                                                       self.y: batch_y, 
                                                                                       self.keep_prob: 1.0})
                validacc = self.sess.run(self.accuracy, feed_dict={self.x: self.valid_data, 
                                                                   self.y: self.valid_labels, 
                                                                   self.keep_prob: 1.0})
                print ("Epoch {0}/{1}, Step {2}/{3}: Learn Rate={4}, Minibatch Loss={5}, Training Accuracy={6}, Validation Accuracy={7}".format(cur_epoch,epochs,step,steps,self.cur_learn_rate,lossval,accval,validacc))
                
                #Update current epoch and learn rate
                cur_epoch = step//steps_per_epoch + 1
                if cur_epoch > 1:
                    self.updateLearnRate(cur_epoch-1)

                #Early stopping policy
                if earlyStopProc:
                    if validacc > 0.993:
                        print("validacc > 0.993 in latest 500. Early Stopping !")
                        break
                    noImproveCount += 1
                    if noImproveCount > 5:
                        noImproveCount = 0
                        earlyStopProc = False
                if validacc > 0.993 and cur_epoch > 100:
                    earlyStopProc = True

                #skip already broken one
                if cur_epoch > 20 and validacc < 0.9:
                    print("Broken. Stop!")
                    break
        print ("Trainning Finished!")
        coord.request_stop()
        coord.join(threads) 
        return earlyStopProc

    def predictTestData(self):
        self.pred_number = self.sess.run(tf.argmax(self.prediction,1), 
                                         feed_dict={self.x:self.test_data, self.keep_prob: 1.0})

    def saveToCSV(self,filestr):    
        with open(filestr,'w') as f:
            id_n = 0
            f.write("ImageId,Label")
            for num in self.pred_number:
                id_n = id_n + 1
                f.write("\n"+str(id_n)+","+str(num))
        print ("Saved to {0}".format(filestr) )       
    


if __name__ == '__main__':
    for loop in range(0,100):
        for i in range(0,4):
            print "i+1={0}".format(i+1)
            recognizer = digitRecognizer()
            recognizer.initDataFromCSV('train.csv', t='train')
            recognizer.initDataFromCSV('test.csv', t='test')
            recognizer.splitTrainValidation(i)
            recognizer.normalizeTrainValidTestData()
            recognizer.augTrainData(5)
            recognizer.createNetTensors()
            recognizer.initSession()
            isEarlyStop = recognizer.train(200)    
            recognizer.predictTestData()
            recognizer.closeSession()
            print "i+1={0}".format(i+1)
            if isEarlyStop:
                recognizer.saveToCSV('[loop{0}]new_results_5_100[{1}].csv'.format(loop,i+1))
            else:
                recognizer.saveToCSV('[loop{0}]new_results_5_100[{1}]_bad.csv'.format(loop,i+1))
            tf.reset_default_graph()
            del recognizer
    
        
        
