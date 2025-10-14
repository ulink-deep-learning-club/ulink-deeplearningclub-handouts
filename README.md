# Deep Learning Club Lecture Materials

This repository contains comprehensive lecture materials and slides for the Ulink Deep Learning Club, designed to provide students with a solid foundation in deep learning concepts and practical implementations.

## 📚 Lecture Series

### L2: Computational Graphs, Backpropagation, and Gradient Descent
- **Topic**: Introduction to the fundamental concepts behind neural network training
- **Content**:
  - Computational graphs as a framework for representing mathematical operations
  - Forward propagation and backward propagation algorithms
  - Gradient descent optimization for parameter updates
  - Practical examples with step-by-step mathematical derivations
- **Format**: Beamer presentation with visual diagrams and mathematical notation
- **Files**: [`presentation.tex`](L2-ComputationalGraph-BackPropagation-GradientDescent/presentation.tex), [`presentation.pdf`](L2-ComputationalGraph-BackPropagation-GradientDescent/presentation.pdf)

### L4: MNIST Digit Recognition - From Fully Connected Networks to CNN
- **Topic**: Comparative analysis of neural network architectures for image classification
- **Content**:
  - Introduction to MNIST dataset and its historical significance
  - Neural network training fundamentals (loss functions, optimization, regularization)
  - Fully connected networks: architecture, implementation, and limitations
  - Convolutional Neural Networks (CNN): principles and advantages
  - Detailed LeNet-5 architecture analysis with PyTorch implementation
  - Performance comparison and architectural insights
  - Neural network scaling laws and modern developments
- **Format**: Comprehensive article with mathematical derivations, code examples, and visualizations
- **Files**: [`mnist.tex`](L4-MNIST/mnist.tex), [`mnist.pdf`](L4-MNIST/mnist.pdf)

## 👥 Target Audience

These materials are designed for:

- **Students with solid math background**: Familiarity with calculus, linear algebra, and probability theory
- **Students with basic Python background**: Understanding of programming fundamentals and basic data structures

## 🎯 Learning Objectives

After studying these materials, students will be able to:

- Understand the mathematical foundations of deep learning
- Implement neural networks from scratch using PyTorch
- Analyze and compare different neural network architectures
- Apply appropriate regularization techniques to prevent overfitting
- Understand the trade-offs between model complexity and performance

## 📖 Pedagogical Approach

Our materials follow these guidelines:

- **Consistency**: Unified notation and terminology throughout all lectures
- **Clarity**: Complex concepts explained with intuitive examples and visualizations
- **Illustrative**: Rich diagrams, code examples, and mathematical derivations
- **Readable**: Well-structured content with clear learning progression
- **Supported by reasons**:
  - Mathematical derivations where applicable
  - Comparative analysis of different approaches
  - Discussion of practical implications and trade-offs

## 🛠 Technical Details

- **Compilation**: Each lecture includes a `compile.sh` script for PDF generation
- **Dependencies**: LaTeX with XeLaTeX compiler, TikZ for diagrams, and standard mathematical packages
- **Language**: Materials are primarily in Chinese with English technical terminology
- **Code Examples**: PyTorch implementations with detailed explanations

## 📁 Repository Structure

```
deep-learning-club-lecture-material/
├── README.md                                    # This file
├── L2-ComputationalGraph-BackPropagation-GradientDescent/
│   ├── presentation.tex                        # Beamer presentation source
│   ├── presentation.pdf                        # Compiled presentation
│   ├── compile.sh                              # Compilation script
│   └── images/                                 # Supporting images
└── L4-MNIST/
    ├── mnist.tex                               # Article source
    ├── mnist.pdf                               # Compiled article
    ├── compile.sh                              # Compilation script
    └── figures/                                # Supporting figures
```

## 🤝 Acknowledgments

These materials were developed with the cooperation of AI and incorporate insights from foundational deep learning research papers.

---

**Last updated**: 2025-10-12
