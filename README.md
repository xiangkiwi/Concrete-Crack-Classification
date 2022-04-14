# Concrete Crack Image Classification in Transfer Learning

The dataset is provided by Medeley: https://data.mendeley.com/datasets/5y9wdsg2zt/2.
The project is built to classify whether the concrete has any crack or not.

## Methodologies
This project is done under Spyder IDE.
For preparing the dataset, AUTOTUNE is run to create prefetch dataset for better performance. Also, RandomFlip in data augmentation is run to increase the training size.

```sh
data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip('horizontal'))
data_augmentation.add(tf.keras.layers.RandomRotation(0.2))
```

This part of coding can be found inside the `concrete_crack_classify.py`.

To train this model, mobilenet_v2 process is used. The epochs is set as 10 to reduce some training time.

```sh
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
```

This part of coding can be found inside the `concrete_crack_classify.py`.

Then, to further improve, fine tuning is run. The epochs is also set for 10 to train the model. 

## Results
After runnning with mobilenet_v2 process, the accuracy is up to 0.9989. 

✨Result after mobilenet_2 process✨

![Mobilenet_init](https://user-images.githubusercontent.com/34246703/163401999-24c17858-b985-455a-a088-d4951a5c697f.PNG)

After running with fine tuning, the accuracy is up to 0.9994, which almost apporaches to 1. 

✨Result after mobilenet_2 process✨

![Mobilenet](https://user-images.githubusercontent.com/34246703/163402030-bb02041e-10e8-491d-8e91-077ff04c8c5f.PNG)

Lastly, some images had been run for prediction. 

✨Prediction and Labels for 4 Sample Images✨

![Prediction vs Label](https://user-images.githubusercontent.com/34246703/163402059-50fe540d-e8a4-4ed8-8b15-6c0a61ec47f3.PNG)

where 0 is negative (no crack) and 1 is positive (crack)

![Prediction Result](https://user-images.githubusercontent.com/34246703/163402096-e71e5725-d17c-42fc-9b55-ca0f329c542b.PNG)


