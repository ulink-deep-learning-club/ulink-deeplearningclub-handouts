# Document Viewer

A modern Vue 3 document viewer with tree-like navigation and iframe display for viewing compiled LaTeX HTML documents.

## Features

- 🌳 Tree-like document navigation
- 🖼️ iframe-based document display
- 📱 Responsive design
- ⚡ Fast Vue 3 Composition API
- 🎨 Modern, clean interface
- 🔄 Document refresh and external link options
- ♿ Accessibility support

## Project Structure

```
Viewer/
├── public/                 # Static assets
├── src/
│   ├── assets/
│   │   └── styles.css     # Global styles
│   ├── components/
│   │   ├── DocumentTree.vue      # Navigation tree component
│   │   └── DocumentViewer.vue    # Main viewer with iframe
│   ├── App.vue            # Main application component
│   └── main.js            # Application entry point
├── dist/                  # Compiled documents and config
│   ├── toc.json          # Table of contents configuration
│   └── [document-folders]/# Compiled HTML documents
├── package.json
└── vite.config.js
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd Viewer
npm install
# or
pnpm install
# or
yarn install
```

### 2. Generate HTML Documents

First, compile your LaTeX documents to HTML using the provided compile script:

```bash
# From the root directory
python compile.py --target html --source-path L2-ComputationalGraph-BackPropagation-GradientDescent --target-path Viewer/dist/L2-ComputationalGraph-BackPropagation-GradientDescent

python compile.py --target html --source-path L4-MNIST --target-path Viewer/dist/L4-MNIST
```

### 3. Configure Table of Contents

Edit `Viewer/dist/toc.json` to include your documents:

```json
{
  "documents": [
    {
      "id": "unique-id",
      "title": "Document Title",
      "path": "folder-name/index.html",
      "description": "Brief description of the document"
    }
  ]
}
```

### 4. Run the Development Server

```bash
npm run dev
# or
pnpm dev
# or
yarn dev
```

The viewer will be available at `http://localhost:3000`

## Building for Production

```bash
npm run build
# or
pnpm build
# or
yarn build
```

The built files will be in the `dist` directory.

## Configuration

### Table of Contents (toc.json)

The `toc.json` file defines the documents available in the viewer:

- `id`: Unique identifier for the document
- `title`: Display title in the navigation
- `path`: Path to the HTML file relative to the `dist` folder
- `description`: Optional description shown in the navigation

### Styling

Global styles are defined in `src/assets/styles.css`. The viewer uses CSS custom properties for theming and supports:

- Light/dark mode (automatic detection)
- High contrast mode
- Reduced motion preferences
- Print styles

## Browser Support

- Chrome/Edge 88+
- Firefox 87+
- Safari 14+
- Mobile browsers

## Development

The project uses:
- Vue 3 with Composition API
- Vite for build tooling
- Modern ES modules

### Key Components

- **App.vue**: Main application layout and state management
- **DocumentTree.vue**: Navigation tree with document selection
- **DocumentViewer.vue**: iframe-based document display with toolbar

## License

MIT