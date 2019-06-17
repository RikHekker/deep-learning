import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
import glob
tf.enable_eager_execution()

def gen(datasettype):
 
  path='/home/kaasbomber/deep-learning/TrainingSetCityScapes/leftImg8bit/train/b*/*' 
  path2='/home/kaasbomber/deep-learning/TrainingSetCityScapes/gtFine/train/b*/*labelIds*'
      
  images_path=glob.glob(path)#glob.glob(path+"/leftImg8bit/train/aachen/*")
  images_path=[x.encode('utf-8') for x in images_path]  
  labels_path=glob.glob(path2)#glob.glob(path+"/gtFine_trainvaltest/gtFine/train/aachen/aachen_[]_gtFine_labellds.png")
  labels_path=[x.encode('utf-8') for x in labels_path]
  couples=zip(images_path,labels_path)
  for paths in couples:
      yield paths   

def read_image_and_label(image_path,label_path):
  #the mapping function, first the paths will be read and the images saved as tensors. Then the tensors will be normalized in [0,1]
  image_tensor=tf.image.decode_image(tf.io.read_file(image_path))
  image_tensor=tf.math.divide(image_tensor,255) #normalize
  #image_tensor = 2*image_tensor-1
  image_tensor = tf.cast(image_tensor, tf.float32)
  image_tensor=tf.image.resize_image_with_crop_or_pad(image_tensor,256,512)
  label_tensor=tf.image.decode_image(tf.io.read_file(label_path))
  label_tensor=tf.image.resize_image_with_crop_or_pad(label_tensor,256,512)
  label_tensor = tf.cast(label_tensor, tf.int32)
  new_cids =[0,0,0,0,0,0,0,1,2,0,0,3,4,5,0,0,0,6,0,7,8,9,10,11,12,13,14,15,16,0,0,17,18,19,0]
  label_new =tf.gather(new_cids, label_tensor)
  return image_tensor,label_new

def input_fn(batch_size_,datasettype):
  #make the dataset usable
  buffer_size_ = 10
  dataset = tf.data.Dataset.from_generator(lambda: gen(datasettype),(tf.string,tf.string))
  dataset = dataset.shuffle(buffer_size=buffer_size_,reshuffle_each_iteration=True).repeat(count=None)
  #.repeat(count=None)
  dataset = dataset.map(read_image_and_label)
  dataset = dataset.batch(batch_size_)
  dataset = dataset.prefetch(None)
  return dataset


def TheModel(batch_size_,restore):
  #build a model and get the loss and accuracy

  dataset=input_fn(batch_size_,datasettype = 'train')
  model=tf.keras.models.Sequential()
  inputs=tf.keras.layers.Input(shape=(256,512,3))
  model.add(ResNet50(include_top=False, weights='imagenet',input_tensor=inputs, pooling=None, classes = 20))
  model.add(tf.keras.layers.Conv2D(20,(3,3)))
  model.add(layers.BatchNormalization())   
  #model.add(layers.Dropout(rate=0.5))
  model.add(tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x,(256,512))))
 
  starter_learning_rate = 0.002
  #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
  #                                         58, 0.96, staircase=True)
  sgd = tf.keras.optimizers.SGD(starter_learning_rate, momentum=0.8, nesterov=True)
  model.compile(optimizer=sgd,   #SGD with momemtum usually gives good enough results without too many parameters as is the case for ADAMoptimizer
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True) #https://gombru.github.io/2018/05/23/cross_entropy_loss/ This site gives good info about how it works it is basically a sofmax function on the groundtruth scores for the classes followed by the cross-entropy loss on the result. The CE simply calculates a score for a ground truths of all classes compared with the log of s ground truth. CE = -log ( \frac{e^{s_{p}}}{\sum_{j}^{C} e^{s_{j}}} 
                ) #accuracy = (tp + tn) / (tp + fp + fn + tn), precision = TP / (TP + FP), recall = TP / (TP + FN) with TP = tf.count_nonzero(predicted * actual) TN = tf.count_nonzero((predicted - 1) * (actual - 1)) FP = tf.count_nonzero(predicted * (actual - 1)) FN = tf.count_nonzero((predicted - 1) * actual), rmse : root of squared differences between scores and ground truths  
  model.fit(dataset, epochs=2, steps_per_epoch=1) #174/6 for Aachen - 2975 Training Images - 1525 Test Images - 500 validation Images
  #history=model.fit(dataset,validation_data=datasetval, epochs=2, steps_per_epoch=103,validation_steps=14) #174/6 for Aachen - 2975 Training Images - 1525 Test Images - 500 validation Images
  saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models/")
  print(saved_model_path)
 

if __name__=='__main__':
  restore = False  
  batch_size_ = 4
  TheModel(batch_size_,restore)
