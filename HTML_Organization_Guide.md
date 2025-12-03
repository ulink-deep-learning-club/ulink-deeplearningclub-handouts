# HTML Organization Guide for Document Viewer

## Current Compilation Process

Based on the `compile.py` script, here's how HTML files are generated:

### Compilation Command
```bash
python compile.py --target html --source-path <document-folder> --target-path <output-folder>
```

### Output Structure
When compiling a LaTeX document to HTML, the script:
1. Creates HTML files with naming pattern `*html.html` 
2. Copies CSS files to the output directory
3. Copies associated directories (images, etc.)
4. Renames the main HTML file to `index.html` if there's only one

### Recommended Organization for Viewer

For the Vue document viewer to work optimally, organize your HTML files as follows:

```
dist/
├── L2-ComputationalGraph-BackPropagation-GradientDescent/
│   ├── index.html          # Main document
│   ├── lwarp.css           # Associated CSS
│   └── images/             # Associated images
├── L4-MNIST/
│   ├── index.html          # Main document  
│   ├── lwarp.css           # Associated CSS
│   └── figures/            # Associated figures
└── toc.json               # Table of contents configuration
```

### Generating HTML Files

To generate HTML files for your documents:

```bash
# For L2-ComputationalGraph-BackPropagation-GradientDescent
python compile.py --target html --source-path L2-ComputationalGraph-BackPropagation-GradientDescent --target-path dist/L2-ComputationalGraph-BackPropagation-GradientDescent

# For L4-MNIST  
python compile.py --target html --source-path L4-MNIST --target-path dist/L4-MNIST
```

### Table of Contents Configuration

Create a `toc.json` file in the `dist` directory with the following structure:

```json
{
  "documents": [
    {
      "id": "l2-computational-graph",
      "title": "L2: Computational Graph, BackPropagation & Gradient Descent",
      "path": "L2-ComputationalGraph-BackPropagation-GradientDescent/index.html",
      "description": "Introduction to computational graphs and optimization"
    },
    {
      "id": "l4-mnist",
      "title": "L4: MNIST Classification",
      "path": "L4-MNIST/index.html", 
      "description": "MNIST digit classification using neural networks"
    }
  ]
}
```

### Viewer Setup

1. Generate HTML files using the compilation script
2. Organize them in the `dist` directory structure shown above
3. Create the `toc.json` configuration file
4. Deploy the Vue viewer application

The viewer will automatically:
- Load the table of contents from `toc.json`
- Display documents in a tree structure on the left
- Show the selected document in an iframe on the right
- Handle navigation between documents