# 4. Tackling Your Own Experiments

In this section, I'm revisiting what we've done thus far and how I recommend approaching new problems with AllenNLP.

## 4.1 Think About the Problem

The first thing we did was figure out how to frame the NER problem --- as either segmentation or token tagging.
Now, in our case the dataset we were using had previously decided this for us.
But often for new tasks in NLP that's not the case.
So it is important to ask questions about what simplifying assumptions you can make, and how to best frame the problem.

When modeling some joint probability distribution, we *have* to make simplifying assumptions to do inference before the heat death of the universe.
In the probability case, we have a structured way of doing that: are these variables independent?
Or can I at least model them as independent without doing too much harm to the model?

In deep learing, how one makes simplifying assumptions can be a bit fuzzier.
For instance, say you're doing document classification.
Do you *need* to have the whole document modeled with BERT (whose text limits are 512)?
Do you even care that the words in the document are in a sequence, and not just a bag-of-words?

There aren't really hard rules for this, but my advice is to think about the simplest solvable version of the problem and attempt to tackle that.
That's why I advocate for building the baselines first.

## 4.2 Read the Data

In this case, read the data means two things.
First, it means that once you're ready to begin tackling a problem in AllenNLP you should write the `DatasetReader` first.
Once you've done this and tested it with the `allennlp train --dry-run` command, you are ready to start building models and experimenting.

But second, it means that you should actually read the data and attempt to solve the task you're about to ask a model to solve.
This is the most valuable advice I got from my undergraduate computer vision course: if you can't solve the problem, why should your model be able to?

When it comes time to actually read the data with AllenNLP, check the [list of pre-written dataset readers](https://allenai.github.io/allennlp-docs/api/allennlp.data.dataset_readers.html) to see if your data is already in a readable format.
You'll notice a CoNLL'03 reader already exists, in fact.
If you were doing NER in real life, and not just a tutorial, I would highly recommend sticking with the AllenNLP reader as opposed to rolling your own.

Another useful question you can ask is: **can I transform my data to be in one of these standard formats?**
Oftentimes, you'll get data in difficult to parse file types (XLSX files, TrecXML files, etc.).
You can speed up your experiments a lot in the future if you just preprocess all of your data into a simple, perhaps even standard, filetype.

If all else fails, I recommend creating a CSV or JSON-lines file for your data.
My own preference is JSON-lines, because it is both easy to read in Python and easy to read for a computer.
Note that a JSON-lines file is a plain text file where each line is a single JSON object with the instance data.

## 4.3 Build the Baselines

Once you have verified that you're reading in the data correctly, I recommend building the baselines.
Oftentimes, these are the quickest models to build.
Perhaps it's just a Seq2Vec encoder (like an LSTM or bag of embeddings encoder) and a Feedforward layer.
The goal here isn't even to get publishable baseline results, it's to ensure that the data loading process went smoothly and your metrics make sense.

## 4.4 Iterate

And, of course, the most enjoyable part of machine learning research is actually doing machine learning research.
Add complexity slowly, **one new element at a time**.

We'll cover this iteration in sections 5-7.

## 4.5 Deploy?

In section 8, we'll extend the 4-part mantra to an optional 5th part: deployment.
I'll walk you through using `Predictor`s, which are a useful way of using AllenNLP to deploy a model.
