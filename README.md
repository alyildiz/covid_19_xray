covid_19_xray
==============================

Chest XRay classification using ResNet152

Classes : Covid, Normal or Viral Pneumonia

If you want to retrain the model download this dataset : https://www.kaggle.com/pranavraikokte/covid19-image-dataset and unzip the file in data/raw

![alt text](https://github.com/alyildiz/covid_19_xray/blob/master/web_app/webapp.jpg?raw=true)

Modeling
==============================
To run the container : ```docker-compose up modeling```
Use ```docker exec -it modeling bash``` to get inside the container. From there, there are 1 endpoint :
- ```/workdir/bin/train_and_save.py``` to train the ResNet152 model (downloading the dataset is mandatory)
You can also visualize the notebook on ```0.0.0.0:8888```

WebApp
==============================
To run the web app : ```docker-compose up web_app```
Go to ```localhost:8501``` to visualize the interface.
 
Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │       └── Covid19-dataset
    │
    │
    ├── modeling           <- Modeling container 
    │
    ├── web_app            <- Web app container
    │
    ├── docker-compose.yml <- to run all containers
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
