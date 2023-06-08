# Training Data

## Download Datasets

Download WoW dataset and place it under `data` directory.

```bash
wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
mkdir wizard_of_wiki
tar -xvf wizard_of_wikipedia.tgz -C wizard_of_wiki/
```

Download Wizint dataset and place it under `data` too.

```bash
wget http://parl.ai/downloads/wizard_of_internet/wizard_of_internet.tgz
mkdir wizard_of_internet
tar -xvf wizard_of_internet.tgz -C wizard_of_internet/
```

## Process Datasets

Use our scripts to preprocess the datasets.

```bash
bash preprocess.sh
```

Note: `nlgeval` is required for preprocessing.