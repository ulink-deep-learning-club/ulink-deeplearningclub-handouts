# Deep Learning Club Lecture Materials

This repository contains lecture materials for the Ulink Deep Learning Club, designed to provide students with a solid foundation in deep learning concepts and practical implementations.

## 📚 Online Documentation

The complete lecture materials are available online at:  
**https://ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts/**

## 📁 Project Structure

- **`source/`** - Main documentation source files in Markdown format
  - `source/index.md` - Main homepage with table of contents
  - `source/lesson2/` - Computational Graphs, Backpropagation, and Gradient Descent
  - `source/lesson4/` - MNIST Digit Recognition: From FC Networks to CNN
  - `source/lesson5/` - UNet: Image Segmentation Architecture
  - `source/lesson6/` - Attention Mechanisms in CNN: From SE-Net to CBAM
  - `source/lesson7/` - PyTorch Basics Tutorial
  - `source/lesson8/` - CNN Ablation Study
  - `source/postscript.md` - Project background and development notes

- **`legacy-doc/`** - Original LaTeX source files (being migrated)
- **`build/`** - Generated HTML documentation
- **`_static/`** - Static assets (images, stylesheets)

## 🚧 Migration Status

**Note**: We are currently migrating from LaTeX to a modern Sphinx-based documentation system. The migration is still in progress, and many TikZ diagrams and images may have rendering issues in the new format.

## 🛠 Build Instructions

To build the documentation locally:

```bash
# Install dependencies
uv sync

# Build HTML documentation
python -m sphinx source build/html

# View the documentation
open build/html/index.html
```

## 📝 About This Project

These materials were created to provide structured, progressive learning resources for deep learning enthusiasts. Rather than using traditional textbooks, we've organized the content as a "knowledge checklist" that guides learners from "what" to "why" to "how".

The materials are designed for students with:
- Solid math background (calculus, linear algebra, probability)
- Basic Python programming knowledge

## 🤝 Contributing

If you find errors or have suggestions for improvement, please feel free to:
1. Fork the repository
2. Make your changes
3. Submit a pull request with detailed description of changes

---

**Last updated**: 2025-12-07