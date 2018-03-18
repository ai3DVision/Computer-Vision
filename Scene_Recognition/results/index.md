# 鄭欽安 <span style="color:red"> (103061148) </span>

# Project 3 / Scene recognition with bag of words

## Overview
The project is related to Scene recognition. This task could be separated to two differnet parts: **Feature extracter and Classifier.** About Feature extracter, we will implement **Tiny Image representation** and **Bag of SIFT representation** to get image feature. About Classifier, we will implement **K-Nearest Neighbor** and **Support Vector Machine** to do classification.     



## Implementation
### Feature extracter
* **Tiny Image representation**
    * Extract image features by resizing each image to 16x16 resolution
    * Apply normalization on them
    * Reshape to vector for each image 
    ```python
    tiny_images = []
    for image_path in image_paths:
        im = Image.open(image_path)
        im = np.array(im.resize((16, 16), Image.BILINEAR))
        im = im - np.mean(im)
        norm = np.linalg.norm(im)
        if norm != 0:
            im = im/norm

        tiny_images.append(im.reshape(-1))
    tiny_images = np.array(tiny_images)
    ```
* **Bag of SIFT representation**
    * **Build vocabulary**
        * Apply SIFT on training data to get SIFT features.
        * Use Kmeans algorithm to cluster them in K (vocabulary size) regions for building vocabulary.
    ```python
    bag_of_features = []
    for path in image_paths:
        img = np.asarray(Image.open(path),dtype='float32')
        frames, descriptors = dsift(img, step=[5,5], fast=True)
        bag_of_features.append(descriptors)
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")        
    ```
    * **Get Bag of SIFT representation**
        * Apply SIFT on input images to get SIFT features.
        * Compare input STFT features to vocabulary STFT features and find the minimum distance from each culsters. Then assign input to its nearest cluster of vocabulary.
        * Use histogram to count how many times each cluster was used
        *  Apply normalization on them
    ```python
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)
    image_feats = []
    for path in image_paths:
        img = np.asarray(Image.open(path),dtype='float32')
        frames, descriptors = dsift(img, step=[5,5], fast=True)
        dist = distance.cdist(vocab, descriptors)
        feat = np.argmin(dist, axis=0)
        feat, bin_edges = np.histogram(feat, bins=range(0, len(vocab)+1))
        norm = np.linalg.norm(feat)
        if norm != 0:
            feat = feat/norm
        image_feats.append(feat)
    image_feats = np.array(image_feats)
    ```
### Classifier
* **K Nearest Neighbor (KNN)**
    * To find which training image feature is closer to testing image feature by calculating distance between them
    * Use voting with top K candidate 
    ```python
    k = 5 
    dict_ = defaultdict(int)
    distance_feats = distance.cdist(test_image_feats, train_image_feats)
    test_predicts = []
    k_idx_lists = np.argsort(distance_feats)[:,:k]
    for k_idx_list in k_idx_lists:
        labels = [ train_labels[idx] for idx in k_idx_list]
        for label in labels:
            dict_[label] += 1
        test_predicts.append(max(dict_.items(), key=operator.itemgetter(1))[0])
        dict_.clear()
    ```
* **Linear Support Vector Machine (SVM)**
    * Apply linear Support Vector Machine to fit training data
    * Use that model for testing prediction
    ```python
    clf = LinearSVC(random_state=0, c=1.0)
    clf.fit(train_image_feats, train_labels)
    pred_label = clf.predict(test_image_feats)
    ```
### Extra
* Try different vocabulary size, such as 100, 400, 1000.
* Non linear SVM
    ```python
    from sklearn.svm import SVC
    clf = SVC(random_state=0, C=1, kernel='rbf')
    clf.fit(train_image_feats, train_labels)
    pred_label = clf.predict(test_image_feats)
    ```

## Installation
* cyvlfeat
```
conda install -c menpo cyvlfeat
```
### Run
```
python proj3.py --feature={tiny_image, bag_of_sift} --classifier={nearest_neighbor, support_vector_machine}
```
## Result
* **K Nearest Neighbor**

||Tiny Image representation|Bag of SIFT representation|
|----|----|----|
|k=1|22.4%|55.33%|
|k=3|22.73%|56.6%|
|k=5|21.8%|56.8%|
|k=7|22.45%|56.53%|

* **Linear Support Vector Machine**

||Tiny Image representation|Bag of SIFT representation|
|----|----|----|
|c=0.01|21.67%|54.13%|
|c=0.1|22.73%|65.6%|
|c=1|21.26%|71.06%|
|c=10|17.87%|69.2%|
|c=100|15.47%|67.2%|

* **Different Vocabulary size (Extra)**
    * Apply KNN and set K = 5
    * Apply Linear SVM and set C = 1

||KNN|Linear SVM|
|----|----|----|
|Vocabulary size = 100|54.2%|65.73%|
|Vocabulary size = 400|56.8%|71.06%|
|Vocabulary size = 1000|56.4%|72.8%|


* **Nonlinear SVM (kernel = rbf) (Extra)**
    * Vocabulary size = 1000
        
||Bag of SIFT representation|
|---|---|
|c=1000|72.33%|

### Confusion matrix

![](./confusion_matrix.png)

### Visualization
| Category name | Sample training images | Sample true positives | False positives with true label | False negatives with wrong predicted label |
| :-----------: | :--------------------: | :-------------------: | :-----------------------------: | :----------------------------------------: |
| Kitchen | ![](thumbnails/Kitchen_train_image_0001.jpg) | ![](thumbnails/Kitchen_TP_image_0192.jpg) | ![](thumbnails/Kitchen_FP_image_0024.jpg) | ![](thumbnails/Kitchen_FN_image_0190.jpg) |
| Store | ![](thumbnails/Store_train_image_0001.jpg) | ![](thumbnails/Store_TP_image_0151.jpg) | ![](thumbnails/Store_FP_image_0026.jpg) | ![](thumbnails/Store_FN_image_0149.jpg) |
| Bedroom | ![](thumbnails/Bedroom_train_image_0001.jpg) | ![](thumbnails/Bedroom_TP_image_0180.jpg) | ![](thumbnails/Bedroom_FP_image_0007.jpg) | ![](thumbnails/Bedroom_FN_image_0175.jpg) |
| LivingRoom | ![](thumbnails/LivingRoom_train_image_0001.jpg) | ![](thumbnails/LivingRoom_TP_image_0147.jpg) | ![](thumbnails/LivingRoom_FP_image_0149.jpg) | ![](thumbnails/LivingRoom_FN_image_0146.jpg) |
| Office | ![](thumbnails/Office_train_image_0002.jpg) | ![](thumbnails/Office_TP_image_0185.jpg) | ![](thumbnails/Office_FP_image_0002.jpg) | ![](thumbnails/Office_FN_image_0140.jpg) |
| Industrial | ![](thumbnails/Industrial_train_image_0002.jpg) | ![](thumbnails/Industrial_TP_image_0152.jpg) | ![](thumbnails/Industrial_FP_image_0001.jpg) | ![](thumbnails/Industrial_FN_image_0144.jpg) |
| Suburb | ![](thumbnails/Suburb_train_image_0002.jpg) | ![](thumbnails/Suburb_TP_image_0176.jpg) | ![](thumbnails/Suburb_FP_image_0081.jpg) | ![](thumbnails/Suburb_FN_image_0053.jpg) |
| InsideCity | ![](thumbnails/InsideCity_train_image_0005.jpg) | ![](thumbnails/InsideCity_TP_image_0134.jpg) | ![](thumbnails/InsideCity_FP_image_0035.jpg) | ![](thumbnails/InsideCity_FN_image_0140.jpg) |
| TallBuilding | ![](thumbnails/TallBuilding_train_image_0010.jpg) | ![](thumbnails/TallBuilding_TP_image_0129.jpg) | ![](thumbnails/TallBuilding_FP_image_0059.jpg) | ![](thumbnails/TallBuilding_FN_image_0131.jpg) |
| Street | ![](thumbnails/Street_train_image_0001.jpg) | ![](thumbnails/Street_TP_image_0147.jpg) | ![](thumbnails/Street_FP_image_0128.jpg) | ![](thumbnails/Street_FN_image_0149.jpg) |
| Highway | ![](thumbnails/Highway_train_image_0009.jpg) | ![](thumbnails/Highway_TP_image_0162.jpg) | ![](thumbnails/Highway_FP_image_0004.jpg) | ![](thumbnails/Highway_FN_image_0144.jpg) |
| OpenCountry | ![](thumbnails/OpenCountry_train_image_0003.jpg) | ![](thumbnails/OpenCountry_TP_image_0125.jpg) | ![](thumbnails/OpenCountry_FP_image_0061.jpg) | ![](thumbnails/OpenCountry_FN_image_0123.jpg) |
| Coast | ![](thumbnails/Coast_train_image_0006.jpg) | ![](thumbnails/Coast_TP_image_0130.jpg) | ![](thumbnails/Coast_FP_image_0060.jpg) | ![](thumbnails/Coast_FN_image_0122.jpg) |
| Mountain | ![](thumbnails/Mountain_train_image_0002.jpg) | ![](thumbnails/Mountain_TP_image_0123.jpg) | ![](thumbnails/Mountain_FP_image_0124.jpg) | ![](thumbnails/Mountain_FN_image_0101.jpg) |
| Forest | ![](thumbnails/Forest_train_image_0003.jpg) | ![](thumbnails/Forest_TP_image_0142.jpg) | ![](thumbnails/Forest_FP_image_0101.jpg) | ![](thumbnails/Forest_FN_image_0128.jpg) |

