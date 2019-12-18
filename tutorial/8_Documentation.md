# 8. Digging into the Documentation

The AllenNLP documentation is some of the best you'll find, of **any open source project, anywhere**.
In this section I want to help you understand how it's organized, how to best make use of it, and how to avoid rewriting code they've already written for you.

## 8.1 Documentation Organization

The core modules that I use are:

- `allennlp.commands` -
- `allennlp.common` -
- `allennlp.data` -
- `allennlp.models` -
- `allennlp.predictors` -
- `allennlp.modules` -
- `allennlp.nn` -
- `allennlp.training` -

**Additional Modules:**

In addition to the above, there are a number of AllenNLP modules that I don't use a lot, but which you might find useful depending on your project.
For completeness' sake, I've listed each of them below with links to the documentation and other resources:

- `allennlp.interpret` -
- `allennlp.semparse` -
- `allennlp.service` -
- `allennlp.state_machines` -
- `allennlp.tools` -

## 8.2 Browsing

## 8.3 Reading the Source

### 8.3.1 Registrables

One of the main reasons to make use of the source is to find the name for a `Registrable`.
