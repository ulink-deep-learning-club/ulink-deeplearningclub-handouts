# Deep Learning Club Lecture Materials

This repository contains comprehensive lecture materials and slides for the Ulink Deep Learning Club, designed to provide students with a solid foundation in deep learning concepts and practical implementations.

## 📚 Lecture Series

### Lesson 2: Computational Graphs, Backpropagation, and Gradient Descent

- **Topic**: Introduction to the fundamental concepts behind neural network training
- **Content**:
  - Computational graphs as a framework for representing mathematical operations
  - Forward propagation and backward propagation algorithms
  - Gradient descent optimization for parameter updates
  - Practical examples with step-by-step mathematical derivations
- **Format**: Beamer presentation with visual diagrams and mathematical notation
- **Files**: [`presentation.tex`](L2-ComputationalGraph-BackPropagation-GradientDescent/presentation.tex), [`presentation.pdf`](dist/L2-ComputationalGraph-BackPropagation-GradientDescent/presentation.pdf)

### Lesson 4: MNIST Digit Recognition - From Fully Connected Networks to CNN

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
- **Files**: [`mnist.tex`](L4-MNIST/mnist.tex), [`mnist.pdf`](dist/L4-MNIST/mnist.pdf)

### Lesson 6: Attention Mechanisms in Convolutional Neural Networks: From SE-Net to CBAM

- **Topic**: Comprehensive exploration of attention mechanisms in CNN architectures
- **Content**:
  - Introduction to attention mechanisms and their biological inspiration
  - Channel attention: Squeeze-and-Excitation Networks (SE-Net) with detailed mathematical analysis
  - Spatial attention: Focus on important spatial locations
  - Hybrid attention: Convolutional Block Attention Module (CBAM) combining channel and spatial attention
  - Theoretical analysis from mathematical, linear algebra, and information theory perspectives
  - Performance comparison and practical implementation guidelines
  - Complete PyTorch implementations with parameter complexity analysis
- **Format**: Comprehensive article with mathematical derivations, code examples, and visualizations
- **Files**: [`document.tex`](L6-AttentionMechanisms/document.tex), [`document.pdf`](dist/L6-AttentionMechanisms/document.pdf)

### Lesson 7: PyTorch Basics Tutorial: From NumPy to Deep Learning

- **Topic**: Comprehensive PyTorch tutorial for beginners with Python background
- **Content**:
  - Transition from NumPy to PyTorch: understanding the necessity of deep learning frameworks
  - Tensors: Core data structure with detailed operations and broadcasting
  - Autograd: Automatic differentiation and computational graphs
  - Neural Network Modules (nn.Module): Building blocks for deep learning models
  - Optimizers: From manual updates to automated optimization algorithms
  - Complete training workflow: Data loading, model definition, training loop, and evaluation
  - Debugging techniques and visualization tools
  - Practical examples with MNIST digit classification
- **Format**: Beginner-friendly tutorial with step-by-step explanations and code examples
- **Files**: [`document.tex`](L7-PyTorch-Basics/document.tex), [`document.pdf`](dist/L7-PyTorch-Basics/document.pdf)

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
- Implement attention mechanisms in CNN architectures
- Master PyTorch framework for deep learning development
- Build complete training pipelines from data loading to model evaluation

## 📖 Pedagogical Approach

Our materials follow these guidelines:

- **Consistency**: Unified notation and terminology throughout all lectures
- **Clarity**: Complex concepts explained with intuitive examples and visualizations
- **Illustrative**: Rich diagrams, code examples, and mathematical derivations
- **Readable**: Well-structured content with clear learning progression
- **Reasonable**:
  - Mathematical derivations where applicable
  - Comparative analysis of different approaches
  - Discussion of practical implications and trade-offs

## 🛠 Technical Details

- **Compilation**: Advanced Python-based build system (`compile.py`) supporting both PDF and HTML output formats
- **Dependencies**: LaTeX with XeLaTeX compiler, TikZ for diagrams, and standard mathematical packages
- **Language**: Materials are primarily in Chinese with English technical terminology
- **Code Examples**: PyTorch implementations with detailed explanations
- **Modular Headers**: Common LaTeX headers extracted into reusable modules for consistency

## 🚀 Build System Usage

### PDF Compilation

```bash
# Compile with hyperref support
python3 compile.py --source-path L4-MNIST/mnist.tex --target-path dist/L4-MNIST/ --hyperref
```

### HTML Compilation

```bash
# Compile to HTML
python3 compile.py --source-path L4-MNIST/ --target-path dist/L4-MNIST/html/ --target html
```

## 🧩 Modular File Headers

The new build system introduces modular LaTeX headers for better maintainability and consistency:

- **`DocumentBaseFormat.tex`**: Base document class, paper size, and font encoding setup
- **`HeaderPackages.tex`**: Common packages (geometry, amsmath, graphicx, tikz, etc.) and custom environments
- **`WebpageHeader.tex`**: HTML-specific headers for web compilation with lwarp

These modules are included in documents using `\input{../Common/...}` commands, allowing centralized management of LaTeX configurations across all lecture materials.

## 📁 Repository Structure

```plaintext
deep-learning-club-lecture-material/
├── README.md                                    # This file
├── compile.py                                   # Advanced Python build system supporting PDF/HTML
├── Common/                                      # Modular LaTeX headers for consistency
│   ├── DocumentBaseFormat.tex                 # Base document class and font setup
│   ├── HeaderPackages.tex                     # Common packages and custom environments
│   ├── WebpageHeader.tex                      # HTML compilation support headers
│   └── DocumentTheme.tex                      # Document theme and styling
├── L2-ComputationalGraph-BackPropagation-GradientDescent/
│   ├── presentation.tex                        # Beamer presentation source
│   ├── presentation.pdf                        # Compiled presentation
│   └── images/                                 # Supporting images
├── L4-MNIST/
│   ├── mnist.tex                               # Article source with modular headers
│   ├── mnist.pdf                               # Compiled article
│   └── figures/                                # Supporting figures
├── L6-AttentionMechanisms/
│   ├── document.tex                            # Attention mechanisms article source
│   ├── document.pdf                            # Compiled attention mechanisms article
│   └── figures/                                # Attention mechanism diagrams
├── L7-PyTorch-Basics/
│   ├── document.tex                            # PyTorch tutorial source
│   ├── document.pdf                            # Compiled PyTorch tutorial
│   └── Assets/                                 # PyTorch tutorial assets
├── Viewer/                                      # Web viewer application
│   ├── src/                                    # Vue.js source code
│   ├── package.json                            # Node.js dependencies
│   └── index.html                              # Main HTML file
└── dist/                                       # Distribution directory for compiled materials
    ├── L2-ComputationalGraph-BackPropagation-GradientDescent/
    ├── L4-MNIST/
    ├── L6-AttentionMechanisms/
    └── L7-PyTorch-Basics/
```

## 🤝 Acknowledgments

These materials were developed with the cooperation of AI and incorporate insights from foundational deep learning research papers.

---

**Last updated**: 2025-11-30
