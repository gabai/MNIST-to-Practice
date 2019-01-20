###
# Title: 
# Objective: Neuralnetwork for eval of medical images
# Author: Gabriel Anaya MD, MAS
###

#Notes: Images are PNG and still pending practice.

library(keras)
library(reticulate)
library(R.utils)

temp <- tempfile() 
urls <- c('https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
            'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
            'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
            'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
            'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
               
            'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
            'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
            'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
            'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
            'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
            'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
            'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz')

names <- C()

for (url in urls) {
  download.file(url, destfile = basename(url))
}

data <- read.csv(untar(temp, "dataset_diabetes/diabetic_data.csv"), header = T, na.strings = c('?','None')) 
unlink(temp)

#Load data
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

#Data Representation
length(dim(train_images)) #axes
dim(train_images) #shape
typeof(train_images) # data type

#Digit example
digit <- train_images[5,,]
plot(as.raster(digit, max = 255))

#Slicing example
my_slice <- train_images[10:99,,]
dim(my_slice)

#NNeural Network
network<- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28*28)) %>%
  layer_dense(units = 10, activation = "softmax")

#Compile
network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#Prep Images
train_images <- array_reshape(train_images, c(60000, 28*28))
train_images <- train_images/255

test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

#Prep Labels
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

#Fit Model
network %>% fit(
  train_images,
  train_labels,
  epochs = 5,
  batch_size = 128
)

#Metrics
metrics<- network %>% evaluate(
  test_images,
  test_labels
)
metrics

#Predictions
network %>% predict_classes(test_images[1:10,])
