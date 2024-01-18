# MLops project : Natural Language Processing
Lanaro Gabriel s233541
Marina Urriola Fernando A s233144
Saccardo Alessia s212246
Suarez Caballero Nerea s233132
Turetta Gabriele s233124


## Overall goal of the project

The final goal of the project is to use natural language processing (NLP)
to create a tranlastor from English to French applying the [Transformer framework](https://huggingface.co/docs/transformers/index).
In order to do this we used [t5-small model](https://huggingface.co/t5-small) using the
[Ted Talks dataset](https://huggingface.co/datasets/ted_talks_iwslt) from [Ted Conference Website](https://ted.com).

## What framework are you going to use and you do you intend to include the framework into your project?
Since the scope of the project is to translate from English to Frencg, we are going to use Transformers framework. Transformers offers pre-built architectures for translation purposes, with a large variety of tools, and we can find it available in PyTorch.
In order to include the framework into our project, we have selected a pre-trained model named T5, which is a text-to-text model that can translate multiple languages among many others tasks available. We will also train, test and validate the model with the selected data.


## What data are you going to run on?

To fine-tune the model, pairs of English-French sentences from The Web Inventory Talk, a collection of the original Ted talks and their translated version, were used. This extensive dataset contains translations in more than 109 languages. An entry in the dataset is in the form:

```txt
{
  "translation": {
    "en": "We need a heroic story for death."
    "fr": "Nous avons besoin d'une histoire héroïque pour la mort.",
  }
}
```


## What models do you expect to use?

The T5 (Text-to-Text Transfer Transformer) model used in our project is a neural network architecture that serves as both an encoder and a decoder. T5 is renowned for transfer learning, being pre-trained on data-rich tasks and subsequently fine-tuned for specific tasks, such as translating between English and French in our case. Its versatility arises from the capability to frame every language problem as a conversion from textual input to textual output, allowing superior performance across a broad spectrum of tasks like summarization, question answering, and text classification. T5 models are available in various sizes, including t5-small, t5-base, t5-large, t5-3b, and t5-11b, catering to different computational needs and task complexities. For the purpose of our project, we opted for t5-small, which encompasses 60 million parameters.



## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
