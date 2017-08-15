import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util
from inference import inference_model
from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
import time
import argparse
import wandb
import keras

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('name', help='name of the run')

parser.add_argument("--load-model", help="load previously saved h5 model")

args = parser.parse_args()
run_name = args.name

conf = wandb.Config(args)

wandb.sync(run_name)




def train(batch_size, epochs, lr_base, lr_power, weight_decay, classes,
          model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=None, batchnorm_momentum=0.9,
          load_model=False, class_weight=None, dataset='VOC2012',
          loss_fn = softmax_sparse_crossentropy_ignoring_last_label,
          metrics = [sparse_accuracy_ignoring_last_label],
          loss_shape=None,
          label_suffix='.png',
          data_suffix='.jpg',
          ignore_label=255,
          label_cval=255):
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)
    batch_shape = (batch_size,) + input_shape

    ###########################################################
    current_dir = os.path.dirname(os.path.realpath(__file__))
    #save_path = os.path.join(current_dir, 'Models/' + model_name)
    save_path = os.path.join(current_dir, run_name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    with open(os.path.join(save_path, 'conf.json'), 'w') as outfile:
        json.dump(conf, outfile)

    # ###############learning rate scheduler####################
    def lr_scheduler(epoch, mode='power_decay'):
        '''if lr_dict.has_key(epoch):
            lr = lr_dict[epoch]
            print 'lr: %f' % lr'''

        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1

        print('lr: %f' % lr)
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)

    # ###################### make model ########################
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')

    model = globals()[model_name](weight_decay=weight_decay,
                                  input_shape=input_shape,
                                  batch_momentum=batchnorm_momentum,
                                  classes=classes)

    # ###################### optimizer ########################
    optimizer = SGD(lr=lr_base, momentum=0.9)
    # optimizer = Nadam(lr=lr_base, beta_1 = 0.825, beta_2 = 0.99685)


    csv_logger = CSVLogger(
            #'output/{}_fcn_vgg16.csv'.format(datetime.datetime.now().isoformat()))
            '%s/tmp_fcn_vgg16_cosmic_small.csv' % run_name)


    with open(conf.test_file_path) as f:
        test_image_list = f.read().splitlines()
        
    #print(test_image_list)
    
    class SaveSampleImages(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.batchnum=0
                
        def on_epoch_end(self, batch, logs={}):
            directory = '%s/samples/%i' % (run_name, self.batchnum)
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_name = 'AtrousFCN_Resnet50_16s'
            #model_name = 'Atrous_DenseNet'
            #model_name = 'DenseNet_FCN'
            weight_file = 'checkpoint_weights.hdf5'
            image_size = (320, 320)
            data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
            label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')


          
            results = inference_model(self.model, image_size, test_image_list, data_dir, label_dir, run_name)
                                    
            for image, result in zip(test_image_list, results):                
                result.save('%s/%s.png' % (directory, image))
                        
            self.batchnum+=1

    
    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=metrics)
    if load_model:
        model.load_weights(load_model)
    model_path = os.path.join(save_path, "model.json")
    # save model structure
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close()
    
    img_path = os.path.join(save_path, "model.png")
    # #vis_util.plot(model, to_file=img_path, show_shapes=True)
    model.summary()

    # lr_reducer      = ReduceLROnPlateau(monitor=softmax_sparse_crossentropy_ignoring_last_label, factor=np.sqrt(0.1),
    #                                     cooldown=0, patience=15, min_lr=0.5e-6)
    # early_stopper   = EarlyStopping(monitor=sparse_accuracy_ignoring_last_label, min_delta=0.0001, patience=70)
    # callbacks = [early_stopper, lr_reducer]
    callbacks = [scheduler]

    # ####################### tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'), histogram_freq=10, write_graph=True)
        callbacks.append(tensorboard)
    # ################### checkpoint saver#######################

    callbacks.append(csv_logger)
    
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.hdf5'), save_weights_only=True)#.{epoch:d}
    callbacks.append(checkpoint)

    save_sample_images = SaveSampleImages()
    callbacks.append(save_sample_images)
    
    # set data generator and train
    train_datagen = SegDataGenerator(zoom_range=[0.5, 2.0],
                                     zoom_maintain_shape=True,
                                     crop_mode='random',
                                     crop_size=target_size,
                                     # pad_size=(505, 505),
                                     rotation_range=0.,
                                     shear_range=0,
                                     horizontal_flip=True,
                                     channel_shift_range=20.,
                                     fill_mode='constant',
                                     label_cval=label_cval)
    val_datagen = SegDataGenerator()

    def get_file_len(file_path):
        fp = open(file_path)
        lines = fp.readlines()
        fp.close()
        return len(lines)
    history = model.fit_generator(
        train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            classes=classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=True,
            loss_shape=loss_shape,
            ignore_label=ignore_label,

            # save_to_dir='Images/'
        ),
        2000,#get_file_len(train_file_path),
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        # validation_data=val_datagen.flow_from_directory(
        #     file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
        #     label_dir=label_dir, label_suffix='.png',classes=classes,
        #     target_size=target_size, color_mode='rgb',
        #     batch_size=batch_size, shuffle=False
        # ),
        # nb_val_samples = 64
        class_weight=class_weight
       )

    model.save_weights(save_path+'/model.hdf5')

if __name__ == '__main__':
    model_name = 'AtrousFCN_Resnet50_16s'
    #model_name = 'Atrous_DenseNet'
    #model_name = 'DenseNet_FCN'
    
    batchnorm_momentum = 0.95
    
    lr_base = 0.01 * (float(conf.batch_size) / 16)
    lr_power = 0.9
    resume_training = False
    if conf.model_name is 'AtrousFCN_Resnet50_16s':
        weight_decay = 0.0001/2
    else:
        weight_decay = 1e-4
    target_size = (320, 320)
    #dataset = 'VOC2012_BERKELEY'
#    if dataset == 'VOC2012_BERKELEY':
        # pascal voc + berkeley semantic contours annotations
        #train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
#        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
#        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
#        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
#        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
#        data_suffix='.jpg'
#        label_suffix='.png'
#        classes = 21
#    if dataset == 'COCO':
        # ###################### loss function & metric ########################
#        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
#        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
#        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
#        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
#        loss_fn = binary_crossentropy_with_logits
#        metrics = [binary_accuracy]
#        loss_shape = (target_size[0] * target_size[1] * classes,)
#        label_suffix = '.npy'
#        data_suffix='.jpg'
#        ignore_label = None
#        label_cval = 0


    # ###################### loss function & metric ########################
    #if dataset == 'VOC2012' or dataset == 'VOC2012_BERKELEY':
    loss_fn = softmax_sparse_crossentropy_ignoring_last_label
    metrics = [sparse_accuracy_ignoring_last_label]
    loss_shape = None
    ignore_label = 255
    label_cval = 255

    # Class weight is not yet supported for 3+ dimensional targets
    # class_weight = {i: 1 for i in range(classes)}
    # # The background class is much more common than all
    # # others, so give it less weight!
    # class_weight[0] = 0.1
    class_weight = None

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    
    
    train(conf.batch_size, conf.epochs, lr_base, lr_power, weight_decay, conf.classes, conf.model_name, conf.train_file_path, conf.val_file_path,
          conf.data_dir, conf.label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum, load_model=conf.load_model,
          class_weight=class_weight, loss_fn=loss_fn, metrics=metrics, loss_shape=loss_shape, data_suffix=conf.data_suffix,
          label_suffix=conf.label_suffix, ignore_label=ignore_label, label_cval=label_cval)
