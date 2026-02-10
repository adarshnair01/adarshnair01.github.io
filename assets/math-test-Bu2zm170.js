const n=`---
title: "The Math Behind Machine Learning"
date: "2024-02-12"
excerpt: "Exploring the fundamental equations that power data science."
author: "Adarsh Nair"
---
# Mathematical Foundations

Data science relies heavily on mathematical concepts. Here are a few key examples formatted beautifully in this journal.

## Linear Regression

The simple linear regression model is defined as:

$$ Y = \\beta_0 + \\beta_1 X + \\epsilon $$

Where:
- $Y$ is the dependent variable
- $X$ is the independent variable
- $\\beta_0$ is the intercept
- $\\beta_1$ is the slope
- $\\epsilon$ is the error term

## The Normal Distribution

The probability density function of the normal distribution is given by:

$$ f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2} $$

## Neural Network Activation

The Sigmoid activation function, commonly used in neural networks:

$$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$

## Matrix Multiplication

Let $A$ be an $m \\times n$ matrix and $B$ be an $n \\times p$ matrix:

$$ (AB)_{ij} = \\sum_{k=1}^n A_{ik}B_{kj} $$
`;export{n as default};
