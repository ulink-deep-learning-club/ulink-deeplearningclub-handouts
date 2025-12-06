# Deep Learning Club Lecture Materials

This repository contains comprehensive lecture materials for the Ulink Deep Learning Club, designed to provide students with a solid foundation in deep learning concepts and practical implementations.

## 🎯 **New: Modern Sphinx Documentation System**

We've migrated from complex LaTeX workflows to a modern Sphinx-based documentation system that provides:

- **📖 Unified Book Format**: All lectures organized as a cohesive book
- **🌐 Responsive HTML**: Clean, searchable web documentation
- **📱 Mobile Friendly**: Works on all devices
- **🔍 Full-text Search**: Built-in search functionality
- **📄 PDF Export**: Generate printable PDF versions
- **🎨 Modern Design**: Sphinx Book Theme with professional styling
- **🔧 Simplified Workflow**: No more complex LaTeX compilation

## 📚 Lecture Series

### 📖 **Complete Sphinx Book** (Recommended)
- **Format**: Modern Sphinx documentation with MyST Markdown
- **Features**: Interactive navigation, search, code highlighting, math support
- **Access**: [`docs/index.md`](docs/index.md) or built HTML at `dist/sphinx/html/`

### Legacy LaTeX Materials (Being Migrated)

#### Lesson 2: Computational Graphs, Backpropagation, and Gradient Descent
- **Topic**: Introduction to the fundamental concepts behind neural network training
- **Content**: Computational graphs, forward/backward propagation, gradient descent
- **New Format**: [`docs/lesson2/index.md`](docs/lesson2/index.md)
- **Legacy Files**: [`presentation.tex`](L2-ComputationalGraph-BackPropagation-GradientDescent/presentation.tex)

#### Lesson 4: MNIST Digit Recognition - From Fully Connected Networks to CNN
- **Topic**: Comparative analysis of neural network architectures for image classification
- **Content**: MNIST dataset, FC networks vs CNN, LeNet-5 implementation
- **New Format**: [`docs/lesson4/index.md`](docs/lesson4/index.md)
- **Legacy Files**: [`mnist.tex`](L4-MNIST/mnist.tex)

#### Lesson 5: UNet - Image Segmentation Architecture
- **Topic**: Encoder-decoder architecture for image segmentation
- **Content**: UNet architecture, skip connections, medical imaging applications
- **New Format**: [`docs/lesson5/index.md`](docs/lesson5/index.md)

#### Lesson 6: Attention Mechanisms in CNN: From SE-Net to CBAM
- **Topic**: Comprehensive exploration of attention mechanisms
- **Content**: Channel attention (SE-Net), spatial attention, hybrid attention (CBAM)
- **New Format**: [`docs/lesson6/index.md`](docs/lesson6/index.md)
- **Legacy Files**: [`document.tex`](L6-AttentionMechanisms/document.tex)

#### Lesson 7: PyTorch Basics Tutorial
- **Topic**: Comprehensive PyTorch tutorial for beginners
- **Content**: Tensors, autograd, nn.Module, training workflow, debugging
- **New Format**: [`docs/lesson7/index.md`](docs/lesson7/index.md)
- **Legacy Files**: [`document.tex`](L7-PyTorch-Basics/document.tex)

#### ✅ **Lesson 8: CNN Ablation Study** (Fully Migrated!)
- **Topic**: Understanding CNN components through systematic ablation studies
- **Content**: Baseline CNN, component analysis, PyTorch implementations, results
- **New Format**: Complete Sphinx documentation in [`docs/lesson8/`](docs/lesson8/)
- **Features**: Split into logical chapters with code examples and tables

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

## 🛠 **New: Modern Documentation System**

### Technical Details

- **📖 Documentation Engine**: Sphinx with MyST Markdown parser
- **🎨 Theme**: Sphinx Book Theme (responsive, modern design)
- **🔧 Extensions**: 
  - `sphinxcontrib.tikz` - Preserves all TikZ diagrams
  - `sphinx_design` - Enhanced components and layouts
  - `myst_parser` - Markdown with LaTeX math support
  - `sphinx.ext.mathjax` - Beautiful math rendering
- **🌐 Output Formats**: HTML (responsive), PDF (printable), ePub
- **🔍 Features**: Full-text search, cross-references, code highlighting
- **📱 Mobile Support**: Fully responsive design works on all devices
- **🔤 Language**: Chinese content with English technical terms
- **💻 Code Examples**: PyTorch implementations with syntax highlighting

### 🚀 **New Build System Usage**

#### Quick Start
```bash
# Install dependencies (uses uv)
uv sync

# Build HTML documentation
python build_sphinx.py --format html --output-dir dist

# Build and serve locally
python build_sphinx.py --format html --serve

# Build all formats (HTML + PDF)
python build_sphinx.py --format all --output-dir dist
```

#### Advanced Usage
```bash
# Clean build (remove previous outputs)
python build_sphinx.py --format html --clean

# Custom output directory
python build_sphinx.py --format html --output-dir docs/_build

# Convert LaTeX to Markdown (for migration)
python convert_latex_to_md.py
```

### 📖 **Content Management**

- **Source Files**: MyST Markdown (`.md`) in `docs/` directory
- **Math Support**: LaTeX math syntax with `$...$` (inline) and `$$...$$` (display)
- **Diagrams**: TikZ code preserved via `sphinxcontrib.tikz` extension
- **Code Blocks**: Syntax highlighting for Python, Bash, etc.
- **Admonitions**: Notes, warnings, tips using MyST directives
- **Tables**: Markdown tables with enhanced styling
- **Cross-references**: Link between lessons and sections

## 🧩 **Legacy LaTeX System** (Being Phased Out)

> **Note**: The complex LaTeX/lwarpmk system is being replaced. Old files remain for reference.

### Old Build System
- **`compile.py`**: Complex LaTeX → HTML/PDF compilation
- **`build_docs.py`**: Batch processing and viewer updates
- **Dependencies**: XeLaTeX, lwarpmk, Vue.js viewer
- **Issues**: Slow compilation, complex workflow, maintenance overhead

### Modular Headers (Legacy)
- **`DocumentBaseFormat.tex`**: Base document class setup
- **`HeaderPackages.tex`**: Common LaTeX packages
- **`WebpageHeader.tex`**: HTML compilation headers
- **Status**: These are being migrated to Sphinx configuration

## 📁 Repository Structure

### 🆕 **New Sphinx-Based Structure**
```plaintext
deep-learning-club-lecture-material/
├── README.md                                    # This file (updated!)
├── pyproject.toml                              # Python dependencies (uv)
├── build_sphinx.py                             # 🆕 Simple Sphinx build system
├── convert_latex_to_md.py                      # 🆕 LaTeX to Markdown converter
├── docs/                                       # 🆕 Sphinx documentation source
│   ├── conf.py                                 # Sphinx configuration
│   ├── index.md                                # Main index with TOC
│   ├── lesson2/                                # Computational Graphs
│   │   └── index.md
│   ├── lesson4/                                # MNIST & CNN
│   │   └── index.md
│   ├── lesson5/                                # UNet
│   │   └── index.md
│   ├── lesson6/                                # Attention Mechanisms
│   │   └── index.md
│   ├── lesson7/                                # PyTorch Basics
│   │   └── index.md
│   ├── lesson8/                                # ✅ CNN Ablation Study (fully migrated)
│   │   ├── index.md
│   │   ├── introduction.md
│   │   ├── experiment-design.md
│   │   └── implementation.md
│   └── _static/                                # Static assets
├── dist/                                       # Distribution directory
│   └── sphinx/                                 # 🆕 Sphinx-built documentation
│       ├── index.html                          # Distribution landing page
│       ├── document.pdf                        # PDF version (if built)
│       └── html/                               # HTML documentation
│           ├── index.html                      # Main documentation
│           ├── lesson2/index.html
│           ├── lesson4/index.html
│           ├── lesson5/index.html
│           ├── lesson6/index.html
│           ├── lesson7/index.html
│           └── lesson8/index.html
└── .venv/                                      # Python virtual environment
```

### 📜 **Legacy Structure** (Being Migrated)
```plaintext
deep-learning-club-lecture-material/
├── compile.py                                  # ❌ Old LaTeX build system
├── build_docs.py                               # ❌ Old batch processor
├── Common/                                     # ❌ Legacy LaTeX headers
│   ├── DocumentBaseFormat.tex
│   ├── HeaderPackages.tex
│   ├── WebpageHeader.tex
│   └── DocumentTheme.tex
├── L2-ComputationalGraph-BackPropagation-GradientDescent/
│   ├── presentation.tex                        # Legacy LaTeX source
│   └── images/
├── L4-MNIST/
│   ├── mnist.tex                               # Legacy LaTeX source
│   └── figures/
├── L5-UNet/                                    # Lesson 5 materials
├── L6-AttentionMechanisms/
│   ├── document.tex                            # Legacy LaTeX source
│   └── figures/
├── L7-PyTorch-Basics/
│   ├── document.tex                            # Legacy LaTeX source
│   └── Assets/
├── L8-CNN-AblationStudy/                       # ✅ Source for migrated lesson
│   ├── document.tex                            # Original LaTeX
│   ├── appendix_template.tex
│   └── Code/                                   # PyTorch code examples
├── Viewer/                                     # ❌ Old Vue.js viewer
│   ├── src/
│   ├── package.json
│   └── index.html
└── temp/                                       # ❌ Temporary build files
```

## 📊 Migration Status

### ✅ **Completed**
- **Sphinx Infrastructure**: Configuration, theme, build system
- **Lesson 8**: CNN Ablation Study fully migrated with code examples
- **Build System**: `build_sphinx.py` replaces complex LaTeX compilation
- **Converter**: `convert_latex_to_md.py` for migrating remaining lessons

### 🔄 **In Progress**
- **Lesson 2**: Computational Graphs (placeholder created)
- **Lesson 4**: MNIST & CNN (placeholder created)
- **Lesson 5**: UNet (placeholder created)
- **Lesson 6**: Attention Mechanisms (placeholder created)
- **Lesson 7**: PyTorch Basics (placeholder created)

### 📋 **Next Steps**
1. Convert remaining LaTeX lessons using the converter script
2. Add TikZ diagram support for migrated content
3. Enhance cross-references between lessons
4. Add interactive examples and quizzes
5. Deploy to GitHub Pages or Read the Docs

## 🎯 **Why We Migrated**

### ❌ **Old System Problems**
- Complex LaTeX compilation with lwarpmk and XeLaTeX
- Separate Vue.js viewer application
- Difficult to maintain and extend
- No search functionality
- Poor mobile experience
- Slow build times

### ✅ **New System Benefits**
- **Simple**: Markdown files are easy to edit
- **Modern**: Responsive design with search
- **Fast**: Instant previews, quick builds
- **Standard**: Uses widely-adopted Sphinx ecosystem
- **Extensible**: Easy to add new features
- **Preserved**: All TikZ diagrams supported via extension

## �� **Getting Started for Contributors**

### For New Content
```bash
# 1. Create new lesson directory
mkdir docs/lesson9

# 2. Create index.md with MyST Markdown
# 3. Add to docs/index.md table of contents
# 4. Build and test
python build_sphinx.py --format html --serve
```

### For Migrating LaTeX
```bash
# 1. Use the converter
python convert_latex_to_md.py

# 2. Manual cleanup and enhancement
# 3. Add MyST directives for better formatting
# 4. Test build
```

## 🤝 Acknowledgments

These materials were developed with the cooperation of AI and incorporate insights from foundational deep learning research papers.

### 🛠 **Migration Credits**
- **Sphinx Infrastructure**: Modern documentation system setup
- **TikZ Preservation**: `sphinxcontrib.tikz` extension integration
- **Build System**: Simplified Python-based build pipeline
- **Content Migration**: LaTeX to MyST Markdown conversion tools

---

**Last updated**: 2025-12-05  
**Migration Version**: 1.0 - Sphinx-based system established
