# 0. Getting Started

In this step we'll learn what AllenNLP is, what the takeaway from this tutorial is, and how to set it up on your machine.

## 0.1 What is AllenNLP

[AllenNLP](https://allennlp.org/ "AllenNLP Homepage") is a library created by the Allen Institute for Artificial Intelligence (AI2), which **makes deep learning for NLP much easier**.
It does this by taking care of most of the boilerplate that people write when doing deep learning (training, early stopping, etc.) with a special focus on boilerplate that people write when doing deep learning for NLP (sequence padding, text loaders, etc.).
It's an incredibly powerful library that improves on the predecessor [TorchText](https://github.com/pytorch/text) in many, many ways, such as:

1. A focus on **experiment repeatability**.
2. **Quick iterations** over architectures or hyperparameters.
3. Easy transitions from **research code to "production" code**.

There are two ways to use AllenNLP: as a library (like, a normal Python library), and as an experimental platform (using JSON configuration files).
The latter is where the real benefits of repeatability, quick iterations, and easy transitions to production code come in.

## 0.2 What will I learn from this tutorial?

The stated goal of this tutorial is, of course, to teach you how to use AllenNLP.
However, the main goal is something much more than that: I hope that by the end of this tutorial you will feel **prepared to approach nearly any deep learning in NLP problem** in a principled way, and will be able to quickly (and repeatably) iterate on hypotheses.
That's certainly a tall order for a single tutorial, but thankfully most of that is taken care of by the AllenNLP library itself; this tutorial is just here to bridge the gaps.

How do I plan on doing that?
Well, there's **a mantra I want you to follow** when using AllenNLP to do NLP research:

> *First, __think about the problem__. Second, __read the data__. Third, __build the baselines__. Then __iterate__.*

That's the whole tutorial in a nutshell.
You'll note that the tutorial structure mirrors the mantra, and it's a reasonable way to approach most NLP research.

### 0.2.1 The Problem

Specifically, we'll be focusing on how to do **named-entity recognition**.
I'll go into more detail about how we're going to approach this in the next section.
But it might be worthwhile to provide a motivating example here.
Say you have the sentence:

```
Bill Gates and Paul Allen, founders of Microsoft, started selling software in 1975 in New Mexico.
```

There are questions you may want to ask in order to understand the sentence:

- **who** was mentioned in the sentence? (Answer: `Bill Gates` and `Paul Allen`)
- **what organizations** were mentioned? (Answer: `Microsoft`)
- **what locations** were mentioned? (Answer: `New Mexico`)
- **what dates** were mentioned? (Answer: `1975`)

You could think of this as a segmentation problem: you want to segment the sentence into spans mentioning an entity, and spans not mentioning entities.
The approach that we'll take, however, attempts to tag each token in the sentence.
We'll go into more about how and why in `Section 1`.
It constitutes the first part of the mantra: **think about the problem**.

### 0.2.2 But First, a Caveat

This is a very opinionated tutorial.
When writing it, I made many (conscious and unconscious) decisions about what to focus on and what to eschew.
As a result, this tutorial covers how *I* use AllenNLP for my research, and how you can adopt this same approach.
For instance, you'll note that there are no Jupyter Notebooks used in this tutorial.
That's because I believe that some of the best features of AllenNLP (repeatability and quick iterations) exist when you use their JSON approach to configuration.

Hopefully I can convince you that these opinions are reasonable and well-considered, but the approach presented here isn't the only way to use AllenNLP.
When possible, I will attempt to link to outside resources that either give a different perspective or supplement what I couldn't get to in this tutorial.

Finally, I have one strong opinion on tutorials in general.
The code in this repository is already complete; it's what you'll end up with, so you can run any of the mentioned commands using this code.
However, you get the most out of any tutorial if you do the leg-work yourself.
I strongly recommend rewriting all the code yourself, **without copy/pasting**.

### 0.2.3 What Should I Already Know?

Pretty much all of this tutorial requires use of the **Unix command line**, so I'm assuming you have access to one (either Mac OS X, a Linux installation, or a virtual machine), or know how to translate these commands to their Windows equivalents.
I'm also assuming you are reasonably familiar (maybe even comfortable) with **Python**, and have a high-level **understanding of what PyTorch is** and how to use it.
If you're not familiar with PyTorch, I highly recommend doing the [60-minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) before starting this tutorial.

Finally, hopefully you're familiar with type systems and object-orientated programming in Python.
Familiarity in this case means that when you see `int n`, you know that it assigns the numeric type `int` to the variable `n`.
AllenNLP **actually makes use of Python type annotations** for experiment configuration!
(I know, if you're a long-time Pythoner, you're probably just as surprised as I was that Python has type annotations.)

## 0.3 Setup

Not only is this tutorial opinionated, AllenNLP is as well.
To make the most of it, you pretty much have to use the latest version of Python (3.7, though 3.6 optionally works), and Python type annotations.
If you don't have Python 3.7 on your machine and don't want to upgrade your system Python version, don't worry.
We'll be using Conda for this tutorial.

### 0.3.1 Installing Anaconda

The first step, thus, is **installing Conda**.
You may optionally skip this step if you already have it installed.
To do this, we recommend following the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for installation.
Whether you choose Conda or MiniConda is up to you, either works for the purposes of this tutorial.

### 0.3.2 Create a Virtual Environment

Once you have Conda installed, the next step is to **create a Conda virtual environment with the newest Python**.
This will allow us to install the newest version of Python and simultaneously not screw up your system's installed version of Python.
To do this, create a new Conda environment called `tagging`, specifying `python=3.7`:

```
conda create -n tagging python=3.7
```

Once you've completed all the prompts, you should be good to go.
Make sure that you can activate the new environment with:

```
conda activate tagging
```

When you're done with the environment, you can `conda deactivate`.
Just make sure to `conda activate tagging` before continuing on with this tutorial!

### 0.3.3 Install AllenNLP

Once you're in the `tagging` environment, you need to install `allennlp`.
(Stylistic note, when I'm talking about AllenNLP abstractly, I'll refer to it as "AllenNLP". When referring to the Python package, I'll use `allennlp`. I'll probably also get those two mixed up at some points.)
Installation is handled through `pip`:

```
pip install allennlp
```

To **verify that this installed correctly**, you can run:

```
allennlp -h
```

And you should see something like the following:

```
Run AllenNLP

optional arguments:
  -h, --help     show this help message and exit
  --version      show program's version number and exit

Commands:

    configure    Run the configuration wizard.
    train        Train a model.
    evaluate     Evaluate the specified model + dataset.
    predict      Use a trained model to make predictions.
    make-vocab   Create a vocabulary.
    elmo         Create word vectors using a pretrained ELMo model.
    fine-tune    Continue training a model on a new dataset.
    dry-run      Create a vocabulary, compute dataset statistics and other
                 training utilities.
    test-install
                 Run the unit tests.
    find-lr      Find a learning rate range.
    print-results
                 Print results from allennlp serialization directories to the
                 console.
```

### 0.3.4 Set up the Tutorial

Finally, you'll have to set up the actual tutorial folders.
To do this, I recommend navigating to wherever you want the code to live and running the following command:

```
mkdir -p allennlp_tutorial/tagging
cd allennlp_tutorial
```

**That's it!**
You're all set up and ready to start learning how to use AllenNLP to do NLP research!
Or, at least, the ever-expanding subset of NLP research that uses deep learning!

## 0.4 Feedback

I highly recommend submitting feedback on this tutorial once you've gone through it!
The best form of feedback is, of course, a **pull request**!
If you see some part of it that needs rewording or recoding, I encourage you to dig in, fix it up, and submit a pull request!
Barring that, I appreciate emails with feedback.
You can email me at jdbarrow [at] cs.umd.edu (and I'll *generally* respond within a week, though no promises).
