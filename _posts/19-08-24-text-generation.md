---
title: "Text & Lyric Generation with Markov Chains" 
excerpt: Text and lyric generation with basic natural language processing.
tags: [projects]
header:
  teaser: assets/images/posts/britney.png
mathjax: "true"
---


# Find the text generator [here](https://generatetext.herokuapp.com/), and the lyric generator [here](https://generatelyrics.herokuapp.com/). 

It fascinates me how simple weighted probabilities can produce semi-comprehensible text. (Playing with text generation, and then writing fiction, I became aware of how my mind also operates like the Markov chain.)

Based off an [old college assignment](https://www.cs.princeton.edu/courses/archive/fall15/cos126/assignments/markov.html), this project uses basic natural language processing to generate semi-plausible text.

{% include figure image_path="/assets/images/posts/britney.png" alt="text generation with Markov chains for Britney Spears lyrics" caption="Britney Spears text generation, my original inspiration." %}

{% include figure image_path="/assets/images/posts/bostrom.png" alt="text generation with Markov chains for Bostrom's <i> Superintelligence </i>" caption="Text generation for Nick Bostrom's <i> Superintelligence </i>." %}


## Markov Chains

The algorithm divides the input text into k-grams, and for each k-gram, forms a frequency table for the following character. When generating text, the algorithm looks at the current k-gram and selects a random character based off the weighted distribution of the frequency table.

The algorithm incorporates the character into the next k-gram and repeats the process. This algorithm is called a Markov chain and produces long-strings of semi-believable text.

Try it out for yourself with [text](https://generatetext.herokuapp.com/) or [lyrics](https://generatelyrics.herokuapp.com/)


## Project Design

I coded the Markov chain in Python and hosted it on Heroku with the Flask micro web framework. I wrote a CSS/HTML/JavaScript GUI to mimic the effect of the Terminal when originally testing the algorithm. For the animation I used the [Typed.js library](https://github.com/mattboldt/typed.js/).

Because the output was especially amusing for song lyrics, I connected a version to the [RapGenius API](https://genius.com/developers).

For Lyric Generator, you enter the name of an artist. The algorithm scrapes the artist's 10 most popular songs from RapGenius and produces a pseudo-random song.

The GitHub repos for the text and lyric generators are [here](https://github.com/soniajoseph/MarkovLyric) and [here](https://github.com/soniajoseph/MarkovLyric2).

