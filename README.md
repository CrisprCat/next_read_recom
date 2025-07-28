# next_read_recom

## Pre-requisites

- Python > 3.11
- JDK >= 17
- Apache Spark



## How to run

1. Create a python virtual environment
```bash
python3.11 -m venv venv
```

2. Activate the python virtual environment
```bash
source venv/bin/activate
```

3. Install the relevant modules
```bash
pip install -r requirements.txt
```


## Dataset

The dataset used for this recommender can be found [here](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_interactions.csv).

In the current version the data needs to be downloaded locally into the working directory so that it can be used for building the recommendation engine
