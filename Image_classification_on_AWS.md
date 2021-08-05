# Building & Deploying Image Classification in AWS Sagemaker



SageMaker’s image classification algorithm is based upon a special convolutional neural network architecture called a ResNet. Before looking at the application of this algorithm, let’s first explore and understand the ResNet architecture used for image classification.

### ResNet

A ResNet is an architecture that is based on the framework of convolutional neural networks and used for problem statements such as image classification. To understand a ResNet, we must first look at the operation of convolutional neural networks. 


![1](https://user-images.githubusercontent.com/23625821/121798505-0f744100-cc27-11eb-99b9-4b90764e1ceb.png)


A typical CNN consists of the following operations:

1. The first operation is the convolution operation, which is also considered an application of filters. We apply different filters on the image so that we can get different versions of the same image, which helps us understand the image perfectly. But, instead of hard-coding the filters, the values of these filters are learned using the backpropagation approach.

2. The next step is called pooling or subsampling. Here, we reduce the size of the image so that the training time becomes faster. There are different types of pooling approaches such as max- pooling, average-pooling, etc.

3. The previous two processes are repeated multiple times, and then the final pooling operation’s output is given to a fully connected neural network layer. Here the major learning happens, and finally the classification task is done.


A problem with the previous architecture is when the network is made too deep; that’s when the backpropagation process suffers. Inside the backpropagation process
the gradients turn to zero, and hence the learning stops. This phenomenon is called vanishing gradients. Therefore, to solve this issue during a deep CNN training, ResNets come into picture.


![1](https://user-images.githubusercontent.com/23625821/121798599-8ad5f280-cc27-11eb-9e65-c94a8aab6eb7.png)


ResNet’s major key is that it allows the flow of gradients in the backward direction. Also, the inputs are bypassed every two convolutions. These two workarounds in CNNs solve the problem of vanishing gradients.

To learn more about ResNet, please visit https://arxiv.org/pdf/1512.03385.pdf.


## SageMaker Application of Image Classification

For this algorithm, we will be using a dataset called Caltech256. It contains about 30,000 images of 256 object categories. These categories include ak47, grasshopper, bathtub, etc.

We can explore more about this dataset or download the dataset from http://www.vision.caltech.edu/Image_Datasets/Caltech256/.

So, in this section, our task is to create a machine learning algorithm that classifies the image into these 256 categories. We will start by defining our roles, regions, etc., that we have already seen in the previous sections. Next, let’s initialize the Docker container of the image classification algorithm.


```py

training_image = get_image_uri(boto3.Session().region_name, 'image-classification')


```


Already we have these images categorized into train and validation sets. We can use these images directly. We can download the images from here:

For training: http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec.

For validation: http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec. 


Let’s move these images to our S3 bucket. These images are in RecordIO-Protobuf format, as the algorithm expects them in that format only. Let’s create a function this time that uploads files to S3.


```py

def upload_to_s3(channel, file):

    s3 = boto3.resource('s3')
    data = open(file, "rb")
    
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)

```

We will now define the folders inside the bucket where we will save the data.

```py
s3_train_key = "image-classification/train"
s3_validation_key = "image-classification/validation"
```


All that is left is to store the image files in S3.

```py
upload_to_s3(s3_train_key, 'caltech-256-60-train.rec')
upload_to_s3(s3_validation_key, 'caltech-256-60-val.rec')
```


Let’s define the parameters related to the algorithm, which we will use to train the model.

```py

num_layers = "18"
image_shape = "3,224,224"
num_training_samples = "15420"

num_classes = "257"
mini_batch_size =  "64"

epochs = "2"
learning_rate = "0.01"

```

The number of layers define the depth of the network. The image shape is 224×224 with three channels (RGB). The total number of images in the training dataset is 15,420. We have a total of 257 classes, 256 objects, and one extra class for others. We define the batch size of 64, which tells that in one go how many images will enter the network. We define the epochs as 2, which means the model will be trained on the whole training dataset two times. Finally, the learning rate is chosen as 0.1, which will decide the number of steps taken to converge and reach the local minima.

We can now define the algorithm. We have already initialized the container.

```py

s3 = boto3.client('s3')

job_name_prefix = 'imageclassification'

job_name = job_name_prefix + '-' + time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())

training_params = {
    "AlgorithmSpecification": {
        "TrainingImage": training_image,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": 's3://{}/{}/output'.format(bucket, job_name_prefix)
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.p2.xlarge",
        "VolumeSizeInGB": 50
    },
    "TrainingJobName": job_name,
    "HyperParameters": {
        "image_shape": image_shape,
        "num_layers": str(num_layers),
        "num_training_samples": str(num_training_samples),
        "num_classes": str(num_classes),
        "mini_batch_size": str(mini_batch_size),
        "epochs": str(epochs),
        "learning_rate": str(learning_rate)
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 360000
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_train,
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-recordio",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_validation,
                     "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-recordio",
            "CompressionType": "None"
        }
    ]
}

```


The following are some of the unique parameters in this algorithm:

1. ContentType is application/x-recordio. As already mentioned, image classification expects only the RecordIO-Protobuf data format.

2. S3DataDistributionType is fully replicated, which means if we use multiple instances for parallel training, then the dataset will be replicated in all the instances.

3. The instance type we are using is p2.xlarge as image classification expects an instance having a graphics card. Be aware that the p2 and p3 instances are not at all free, and they are chargeable. 


Once we are done with algorithm specifications, we will start the training process.


```py

sagemaker = boto3.client(service_name='sagemaker')

sagemaker.create_training_job(**training_params)

status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']

print(status)

while status !='Completed' and status!='Failed':
    time.sleep(60)
    
    status = client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
    
    print(status)

```

As shown, this code will start the training and then inform us whether the training successfully finished. Once the training is finished, we will deploy the model and then do the predictions. 

```py

endpoint_response= sagemaker.create_endpoint(
                        EndpointName="image-classification-caltech-endpoint", 
                        EndpointConfigName="image-classification-caltech-config")
    
```    
 

This will take some time and spin up the resource required for the endpoint generation. Once the endpoint is generated, we can start the predictions.

Let’s download an image and use it for testing our model. We can download a bathtub image and check whether the model predicts it perfectly.

```sh

! wget -O /tmp/test.jpg http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/008.bathtub/008_0007.jpg

```
The previous line will directly download the image to your system. If your system is not Linux, then you can directly go to the link and download the image. Next, we need to read the image and then pass it to the endpoint.


```py

with open('/tmp/test.jpg', 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
    
    
response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                   ContentType='application/x-image',
                   Body=payload)
                   
                   
result = response['Body'].read()
result = json.loads(result)

```

The variable result consists of probabilities of prediction for all the classes. We need to find the class that has the maximum probability. That means if we can get the argument that has the maximum probability, that will be our predicted class. For this we can use the np.argmax() function.

```py
index = np.argmax(result)
```

Now, we can use this index to extract the label. We can create a list of all the labels in the same sequence as they are present in the dataset, and then we can pass the index to predict the label. Suppose we save all the classes in the variable object_classes. Next we can print the prediction.

```py
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))
```


Don’t forget to delete the endpoint once all the operations are done.

```py
sage.delete_endpoint(EndpointName=endpoint_name)
```
