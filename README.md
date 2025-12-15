# Deep Learning Club Lecture Materials

This repository contains lecture materials for the Ulink Deep Learning Club, designed to provide students with a solid foundation in deep learning concepts and practical implementations.

## 📚 Online Documentation

The complete lecture materials are available online at:  
**https://ulink-deep-learning-club.github.io/ulink-deeplearningclub-handouts/**

## 📁 Project Structure

- **`source/`** - Main documentation source files in Markdown format
  - `source/index.md` - Main homepage with table of contents
  - `source/math-fundamentals/` - Computational Graphs, Backpropagation, Gradient Descent, Loss Functions, Activation Functions
  - `source/neural-network-basics/` - Fully Connected Layers, CNN Basics, LeNet, Neural Training Basics, Scaling Laws
  - `source/attention-mechanisms/` - SE-Net, CBAM, Spatial Attention, Channel Attention, Extensions
  - `source/unet-image-segmentation/` - UNet Architecture, Implementation, Loss Functions, Data Augmentation
  - `source/pytorch-practice/` - PyTorch Basics, Tensor Operations, Autograd, Training Workflow, Best Practices
  - `source/cnn-ablation-study/` - Experiment Design, Implementation, Results Analysis
  - `source/postscript.md` - Project background and development notes

- **`legacy-doc/`** - Original LaTeX source files (archived for reference)
- **`build/`** - Generated HTML documentation
- **`_static/`** - Static assets (images, stylesheets)

## 🚧 Migration Status

**Note**: The migration from LaTeX to a modern Sphinx-based documentation system is almost complete. All lecture materials have been refactored into a modular Markdown structure with improved navigation and code integration. The legacy LaTeX sources are kept in `legacy-doc/` for reference. The migration of TikZ diagrams is incomplete.

## 🛠 Build Instructions

To build the documentation locally:

```bash
# Venv preparation
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# Build HTML documentation
make build

# View the documentation
cd build/html && python3 -m http.server 1200 & open "http://localhost:1200"
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
