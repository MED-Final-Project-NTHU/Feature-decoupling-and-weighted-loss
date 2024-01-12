from tfrecord.torch.dataset import TFRecordDataset
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def generateData(mode, inputname, outputname, seed):

    label_name = ['Atelectasis','Calcification of the Aorta','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Subcutaneous Emphysema','Support Devices','Tortuous Aorta']
    raw_dataset = tf.data.TFRecordDataset(inputname)
    for raw_record in raw_dataset:
        print('ok')

    print(ds)

    if not os.path.exists(f'{mode}_labels.npy'):
        labels = []
        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = dict(example.features.feature.items())
            
            label = []
            for i, name in enumerate(label_name):
                label.append(features[name].float_list.value[0])
            labels.append(label)
        # print('labels fin...')
        np.save(f'{mode}_labels.npy', labels)
    
    labels = np.array(np.load(f'{mode}_labels.npy'))
    print(labels.shape)

    minority_size = np.min(np.sum(labels, axis=0))
    new_labels_indices = []                                         # indices of samples that we want to keep
    label_cnt = np.zeros(19)                                        # currently, the number of samples that has the ith label
    n_samples = 0

    thresholds = int(0.7*minority_size)                             # we want the smallest class to have at least 70% of size of the minority class
    np.random.shuffle(labels)
    for i, label in enumerate(labels):                                
        indices = np.where(labels[i] > 0)[0]                        # in an example, find indices where the value is 1
        if not np.all(label_cnt[indices] > thresholds):             # If the all labels of this samples have enough samples already, then we skip; Otherwise, we add this sample to our dataset
            new_labels_indices.append(i)
            label_cnt += label
            n_samples += 1

    print(f"\nTotal {n_samples} samples in the {mode} dataset")
    print(f"Number of samples contain the ith label: ")
    print(f"Before: {np.sum(labels, axis=0)}")
    print(f"After : {label_cnt}\n")


    # write file
    with tf.io.TFRecordWriter(outputname, options=None) as writer:
        id = -1
        for raw_record in raw_dataset:
            id += 1
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = dict(example.features.feature.items())

            if id in new_labels_indices:
                example = tf.train.Example(features=tf.train.Features(feature=features))
                output_string = example.SerializeToString()
                writer.write(output_string)
        

if __name__ == "__main__":

    inputname = f"../MICCAI_long_tail_train.tfrecords"

    for seed in [0, 1111, 222, 33, 4444]:
        np.random.seed(seed)
        random.seed(seed)

        generateData(mode='train', inputname=inputname, outputname=f"MICCAI_classBalanced_train_{seed}.tfrecords", seed=seed)



# Total 3532 samples in the train dataset
# Number of samples contain the ith label:
# Before: [51055.  2595. 55020. 12406. 30553. 19880.  7823.  1707. 57591. 34376.
#  53029.   373.   571. 32131.   448. 11579.  1900. 73639.  2212.]
# After : [1038.  271. 1049.  394.  558.  710.  355.  273. 1205.  262. 1142.  262.
#   264.  562.  262.  556.  361. 1463.  271.]

# (182380, 19)

# Total 3529 samples in the train dataset
# Number of samples contain the ith label:
# Before: [51055.  2595. 55020. 12406. 30553. 19880.  7823.  1707. 57591. 34376.
#  53029.   373.   571. 32131.   448. 11579.  1900. 73639.  2212.]
# After : [1054.  270. 1040.  381.  531.  734.  348.  276. 1242.  262. 1104.  262.
#   264.  596.  262.  571.  363. 1503.  268.]

# (182380, 19)

# Total 3518 samples in the train dataset
# Number of samples contain the ith label:
# Before: [51055.  2595. 55020. 12406. 30553. 19880.  7823.  1707. 57591. 34376.
#  53029.   373.   571. 32131.   448. 11579.  1900. 73639.  2212.]
# After : [1086.  271. 1021.  375.  559.  727.  341.  275. 1211.  262. 1119.  262.
#   264.  612.  262.  545.  340. 1467.  268.]

# (182380, 19)

# Total 3477 samples in the train dataset
# Number of samples contain the ith label:
# Before: [51055.  2595. 55020. 12406. 30553. 19880.  7823.  1707. 57591. 34376.
#  53029.   373.   571. 32131.   448. 11579.  1900. 73639.  2212.]
# After : [1031.  268. 1012.  404.  531.  723.  358.  277. 1193.  262. 1110.  262.
#   264.  574.  262.  546.  352. 1461.  272.]

# (182380, 19)

# Total 3507 samples in the train dataset
# Number of samples contain the ith label:
# Before: [51055.  2595. 55020. 12406. 30553. 19880.  7823.  1707. 57591. 34376.
#  53029.   373.   571. 32131.   448. 11579.  1900. 73639.  2212.]
# After : [1048.  269. 1013.  385.  523.  731.  360.  274. 1212.  262. 1081.  262.
#   265.  571.  262.  540.  350. 1501.  268.]