# Kaggle Camera

Initial setup:
* Install Git & clone this repository: `sudo apt-get install -y git && git clone https://github.com/marvelousNinja/kaggle-camera`
* In repo folder run: `./setup.sh`
* Configure Kaggle CLI: `kg config -c sp-society-camera-model-identification -u KAGGLE_USERNAME -p KAGGLE_PASSWORD`
* Download competition data: `kaggle download`
* Unzip test and train datasets: `unzip train.zip -d data && unzip test.zip -d data`
