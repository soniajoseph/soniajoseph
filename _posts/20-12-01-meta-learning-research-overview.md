---
title: Meta-learning Research Overview and Paper Group
tags: [meta-learning, machine learning, artificial intelligence, resources]
header: 
  teaser: assets/images/posts/neuralnet.jpg
excerpt: Research overview of meta-learning by researcher and institution.
categories: [machine learning]
toc: true
mathjax: "true"
comments: true
collection: posts
comments: true
classes: wide
---

*Incomplete and in progress. Feel free to comment paper recommendations below.*

# Radical Meta-Learning Paper Group

Created: Jan 1, 2021 7:26 PM
Event Leaders: Sonia Joseph

# Radical Meta-learning

Reading Group Winter 2021

## FAQ

***What is the purpose of this reading group?***

There is a trend for algorithms to move away from handcrafted parameters and toward minimizing the human-specified inductive bias as much as possible. Meta-learning is a continuation of this trend.

In this reading group, we want to gain a deep understanding of “radical” meta-learning, i.e., algorithms that learn how to learn how to learn -- and on till the nth meta-level.

***Why “radical”?***

“Radical” is a word taken from Jurgen Schmidhuber to distinguish this branch of meta-learning from current transfer learning techniques (e.g. MAML). For example, this difference could be between learning gradient descent itself (radical meta-learning) and optimizing gradient descent across many data distributions (a type of meta-learning / transfer learning). We want to study the first camp: learning as much of the algorithm as possible from scratch.

***How do I join?***

Shoot a quick email to Sonia Joseph at smjoseph@alumni.princeton.edu

***What is the format?*** 

One person will sign up to present a paper, which everyone will read. We'll collectively take notes on a Google doc before and during the presentation. The presenter will go through the paper, and we'll liberally discuss along the way. Some of the papers are long, so please feel free to come anyway even if you haven't finished it.

## Phase 2

### Reading list  [AI-GA related]

1. *Generative teaching networks: learning to teach by generating synthetic training data.* [Link](https://arxiv.org/abs/1912.07768)
2. *Differentiable plasticity: training plastic neural networks with backpropagation.* [Link](https://arxiv.org/abs/1804.02464) 
3. *Paired open-ended trailblazer (poet): Endlessly generating increasingly complex and diverse learning environments and their solutions.* [Link](https://arxiv.org/abs/1901.01753)

### Schedule [tentative]

1. **Seminar 1. April 11th, 6:00 pm GMT**
    1. POET by Anushan.
2. **Seminar 2. April 25th, 6:00 pm GMT**
    1. Differentiable Plasticity by Harsha

## Phase 1:

### Sample reading list [please put your name next to what you would like to present]

1. *AIXI. [Link.](https://arxiv.org/abs/cs/0004001)*
2. *Evolutionary principles in self-referential learning. (On learning how to learn: The meta-meta-... hook.). 1987.* [Link](http://people.idsia.ch/~juergen/diploma1987ocr.pdf). **Presenter: Sonia Joseph**
3. *Ultimate Cognition a la Godel. 2009.* [Link](http://people.idsia.ch/~juergen/ultimatecognition.pdf). **Presenter: Sid**
4. *Reinforcement Learning with Self-Modifying Policies. 1997. [Link](http://people.idsia.ch/~juergen/ssabook/ssabook.html).*
5. *Optimally Ordered Problem Solver. 2004. [Link](http://people.idsia.ch/~juergen/oopsweb/oopsweb.html).*
6. *AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence. [Link](https://arxiv.org/pdf/1905.10985.pdf). **Presenter**: Acyr**.*** 
7. *Improving Generalization in Meta Reinforcement Learning using Learned Objectives. 2019.* [Link](https://arxiv.org/abs/1910.04098). **Presenter**: Rob/Louis Kirsch (?)
8. *Discovering Reinforcement Learning Algorithms. 2020. [Link.](https://arxiv.org/abs/2007.08794)* **Presenter**: Rob
9. *Meta-Gradient Reinforcement Learning with an Objective Discovered Online. 2020. [Link](https://arxiv.org/pdf/2007.08433.pdf).* **Presenter**: Rob

### Schedule [tentative]

*We will have four meetings.This is rough--- we’ll finalize the schedule with a whenisgood.*

1. **Seminar 1. Sunday Jan 29, 6:00 pm GMT**
    1. Whenisgood for first meeting: [https://whenisgood.net/78wbi7k](https://whenisgood.net/78wbi7k) 
    2. **Paper**: *Evolutionary principles in self-referential learning. (On learning how to learn: The meta-meta-... hook.). 1987.* [Link](http://people.idsia.ch/~juergen/diploma1987ocr.pdf). 
    3. **Presenter**: Sonia Joseph
        1. **Slides:** [https://docs.google.com/presentation/d/14uhiWYJUOILFybIEKvaiZPs5y8xuF6YgzINygE5E9Ys/edit#slide=id.p](https://docs.google.com/presentation/d/14uhiWYJUOILFybIEKvaiZPs5y8xuF6YgzINygE5E9Ys/edit#slide=id.p)
    4. **Notes**: [https://docs.google.com/document/d/1HBzO4YEm2L8drf82UPxCRDlAqiRzMJcJW77ccyCoLW4/edit?usp=sharing](https://docs.google.com/document/d/1HBzO4YEm2L8drf82UPxCRDlAqiRzMJcJW77ccyCoLW4/edit?usp=sharing)
2. **Seminar 2. Feb 14th, 6:00 pm GMT**
    1. **Paper**: Ultimate Cognition a la Godel. 
    2. **Presenter**: Sid
    3. **Notes**: [https://docs.google.com/document/d/1Wg4Aoy9qhhaBpPrUxp9KNhBf2f3-HwiagCN4arQhJ5E/edit#](https://docs.google.com/document/d/1Wg4Aoy9qhhaBpPrUxp9KNhBf2f3-HwiagCN4arQhJ5E/edit#)
3. **Seminar 3. Feb 26 weekend.**
    1. **Paper**: AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence. [Link](https://arxiv.org/pdf/1905.10985.pdf). 
    2. **Presenter**: Acyr.
    3. **Notes**: *Coming soon.*
4. **Seminar 4. March 12th weekend.**
    1. **Paper:** Meta-Policy Gradients - Papers: 6,7,8.
    2. **Presenter**: Rob.
    3. **Notes**: Checkout this fantastic [blog post](https://roberttlange.github.io/posts/2020/12/meta-policy-gradients/) 👨‍🔧


## Research Papers by Concept

### Proof-Search
* [Ultimate Cognition a la Godel](http://people.idsia.ch/~juergen/ultimatecognition.pdf)(2009)
* [Optically Ordered Problem-Solver](http://people.idsia.ch/~juergen/oopsweb/oopsweb.html) (2004)

### Transfer Learning

### Reinforcement Learning
* [Meta-Gradient Reinforcement Learning with an Online Objective Discovered Online](https://arxiv.org/pdf/2007.08433.pdf) (2020)
* [Discovering Reinforcement Learning Algorithms](https://arxiv.org/pdf/2007.08794.pdf) (2021)
* [Improving Generalization in Meta Reinforcement Learning using Learned Objectives](https://arxiv.org/abs/1910.04098) (2019)

### Generative Models
* [AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence](https://arxiv.org/pdf/1905.10985.pdf) (2020)

## Literature Reviews

## Courses


## Research Papers by Institution


### Berkeley


### DeepMind
1. [Meta-Gradient Reinforcement Learning with an Online Objective Discovered Online](https://arxiv.org/pdf/2007.08433.pdf) (2020)


### IDSIA
1. Jurgen Schmidhuber
	1. [Evolutionary Principles in Self-Referential Learning](http://people.idsia.ch/~juergen/diploma.html) (1987)
	2. [Ultimate Cognition a la Godel](http://people.idsia.ch/~juergen/ultimatecognition.pdf)(2009)
2. Louis Kirsch
	1. [Improving Generalization in Meta Reinforcement Learning using Learned Objectives](https://arxiv.org/abs/1910.04098) (2019)

### MILA

### Stanford
1. Chelsea Finn [from Berkeley]


### University of Oxford
