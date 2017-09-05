import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
from utils.metrics import *
from models import *



def inference_model(model, image_size, image_list, data_dir, label_dir, run_name, return_results=True, save_dir=None,
                            label_suffix='.png',
                            data_suffix='.jpg', target_size=None, log_file=None
):
    batch_shape = (1, ) + image_size + (3, )
    results = []
    total = 0
    img_to_acc = {}
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total,img_num))
        print('%s/%s%s' % (data_dir, img_num, data_suffix))
        image = Image.open('%s/%s%s' % (data_dir, img_num, data_suffix))
        image = image.convert('RGB')
        
        image = img_to_array(image)  # , data_format='default')
        #print("Shape", image.shape)
        label = Image.open('%s/%s%s' % (label_dir, img_num, label_suffix))
        label_size = label.size
        #print("Label size", label.size)
        img_h, img_w = image.shape[0:2]
        
        # long_side = max(img_h, img_w, image_size[0], image_size[1])
        pad_w = max(image_size[1] - img_w, 0)
        pad_h = max(image_size[0] - img_h, 0)
        image = np.lib.pad(image, ((pad_h/2, pad_h - pad_h/2), (pad_w/2, pad_w - pad_w/2), (0, 0)), 'constant', constant_values=0.)
        y = img_to_array(
            label).astype(int)

                                    
        y = img_to_array(label.resize((target_size[1], target_size[
                                                     0]), Image.NEAREST)).astype(int)
        y = y[:,:,0]
#        y = np.lib.pad(y, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2), (0, 0)), 'constant', constant_values=255.)
        
        # image -= mean_value
        '''img = array_to_img(image, 'channels_last', scale=False)
        img.show()
        exit()'''
        image = cv2.resize(image, image_size)

        #print("Image Shape", image.shape)
        #print("Image Size", image_size)

        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        result = model.predict(image, batch_size=1)


        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)
        result_img = Image.fromarray(result, mode='P')
        result_img.palette = label.palette
        # result_img = result_img.resize(label_size, resample=Image.BILINEAR)
        result_img = result_img.crop((pad_w/2, pad_h/2, pad_w/2+img_w, pad_h/2+img_h))
        # result_img.show(title='result')

        acc = sum(y==result)/((y.shape[0]*y.shape[1])+0.)
        img_to_acc[img_num]=acc

        
            
        
        if return_results:
            results.append(result_img)
        if save_dir:
            result_img.save(os.path.join(save_dir, img_num + '.png'))


    if log_file:
        with open(log_file, 'w') as outfile:
            json.dump(img_to_acc, outfile)

    return results


    
def inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, run_name, return_results=True, save_dir=None,
              label_suffix='.png',
              data_suffix='.jpg'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # mean_value = np.array([104.00699, 116.66877, 122.67892])
    batch_shape = (1, ) + image_size + (3, )

    #old save path
    #save_path = os.path.join(current_dir, 'Models/'+model_name)

    save_path = os.path.join(current_dir, run_name)
    model_path = os.path.join(save_path, "model.json")
    checkpoint_path = os.path.join(save_path, weight_file)

    # model_path = os.path.join(current_dir, 'model_weights/fcn_atrous/model_change.hdf5')
    # model = FCN_Resnet50_32s((480,480,3))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    model = globals()[model_name](batch_shape=batch_shape, input_shape=(512, 512, 3))
    model.load_weights(checkpoint_path, by_name=True)

    model.summary()

    results = []
    total = 0
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total,img_num))
        image = Image.open('%s/%s%s' % (data_dir, img_num, data_suffix))
        image = img_to_array(image)  # , data_format='default')

        label = Image.open('%s/%s%s' % (label_dir, img_num, label_suffix))
        label_size = label.size

        img_h, img_w = image.shape[0:2]

        # long_side = max(img_h, img_w, image_size[0], image_size[1])
        pad_w = max(image_size[1] - img_w, 0)
        pad_h = max(image_size[0] - img_h, 0)
        image = np.lib.pad(image, ((pad_h/2, pad_h - pad_h/2), (pad_w/2, pad_w - pad_w/2), (0, 0)), 'constant', constant_values=0.)
        # image -= mean_value
        '''img = array_to_img(image, 'channels_last', scale=False)
        img.show()
        exit()'''
        # image = cv2.resize(image, image_size)

        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        result = model.predict(image, batch_size=1)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

        result_img = Image.fromarray(result, mode='P')
        print("result shape", result_img.shape)
        result_img.palette = label.palette
        # result_img = result_img.resize(label_size, resample=Image.BILINEAR)
        result_img = result_img.crop((pad_w/2, pad_h/2, pad_w/2+img_w, pad_h/2+img_h))
        # result_img.show(title='result')
        if return_results:
            results.append(result_img)
        if save_dir:
            result_img.save(os.path.join(save_dir, img_num + '.png'))
    return results

import wandb
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('run_name', help='path to model')

    args = parser.parse_args()

    conf = wandb.Config(args)

    model_name = 'AtrousFCN_Resnet50_16s'
    #model_name = 'Atrous_DenseNet'
    #model_name = 'DenseNet_FCN'

    target_size = (320, 320)
        
    image_size = (320, 320)
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)
        batch_shape = (conf.batch_size,) + input_shape

    with open(conf.test_file_path) as f:
        test_image_list = f.read().splitlines()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    #save_path = os.path.join(current_dir, 'Models/' + model_name)
    save_path = os.path.join(current_dir, conf.run_name)
        
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')
    
    model_path = os.path.join(save_path, "model.json")
        
    model = globals()[model_name](
                                      input_shape=input_shape,
                                      classes=conf.classes)
    model.load_weights(checkpoint_path, by_name=True)

    model.summary()
        
    results = inference_model(model, image_size, test_image_list, conf.data_dir, conf.label_dir, conf.run_name,
                                          label_suffix=conf.label_suffix, data_suffix=conf.data_suffix,
                              target_size=target_size, log_file='logs.json')
    for result in results:
#        result.show(title='result', command=None)
        result.save('out.png')
