# Speech-Audio-Filter 🔊
![Image](https://github.com/Ashish-Abraham/Audio-Filter/blob/main/Images/richard-horvath-WOA3QKFjlo8-unsplash.jpg)
Noise suppression in human speech audio samples using generative ML model. Identifying a person's audio from a mixed environment input.
## Neural Network
* Encoder-Decoder
* Framework : TensorFlow(2.11.0)
* Dataset : https://datashare.ed.ac.uk/handle/10283/1942 <br><br>

## Model Training
The architecture for a basic model is found in **./base_model.py**. Use below code to create model.
```
model = create_model(batching_size, activation_func)
```
```
model.compile(optimizer, loss)
history = model.fit(train_dataset,epochs)
```
 
## How to use:
* Clone repository to local machine and open **predict.py**<br>
* Use function **predict(audio_path, model_path)**.

## Results
**Input**<br><br>
![Image](https://github.com/Ashish-Abraham/Audio-Filter/blob/main/Images/noisy_speech.png)<br><br>
**Ground Truth**<br><br>
![Image](https://github.com/Ashish-Abraham/Audio-Filter/blob/main/Images/pure_speech.png)<br><br>
**Prediction vs Input**<br><br>
🟠Prediction   🔵Input<br><br>
![Image](https://github.com/Ashish-Abraham/Audio-Filter/blob/main/Images/result.png)<br><br>


## To Do:
* Quantize model using TFLite


