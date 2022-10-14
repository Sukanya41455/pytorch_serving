## Installation
```
sudo apt install --no-install-recommends -y openjdk-11-jre-headless
```
```
pip install torch torchserve torch-model-archiver torch-workflow-archiver
```

## To generate archive torch model
```
./generate_model_archive.sh
```

## Register and serving the model

This will start the model-server in your localhost.
The arguments --model-store is used to specify the location from which the models can be loaded. --models MODEL_NAME=<PATH_TO_MAR_FILE> is used to register the models
--ncs prevents the server from storing config snapshot files. 
```
torchserve --start --model-store model_store --models my_model=BERTweetSentimentAnalysis.mar --ncs
```

The important things to see here is the Inference address, Management address, and Metrics address. These addresses show the URLs that you access to generate predictions, manage models, and see the model metrics, respectively. Take the inference address as ane example, you can send a POST request containing your JSON file to localhost:8080/predictions/my_model 

## To stop model-server
```
torchserve --stop
```