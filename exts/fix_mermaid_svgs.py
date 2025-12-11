"""
Sphinx post-processing script to fix mermaid SVG dimensions.
Removes width="100%" and adds explicit width/height from viewBox.
"""

import re
from pathlib import Path


def fix_mermaid_svg(file_path: Path) -> bool:
    """
    Fix a single mermaid SVG file.
    
    Returns True if file was modified, False otherwise.
    """
    content = file_path.read_text(encoding='utf-8')
    
    # Check if width="100%" exists
    if 'width="100%"' not in content:
        return False
    
    # Extract viewBox values (supports decimals)
    viewbox_pattern = re.compile(r'viewBox="0\s+0\s+([\d.]+)\s+([\d.]+)"')
    viewbox_match = viewbox_pattern.search(content)
    
    if not viewbox_match:
        return False
    
    width = viewbox_match.group(1)
    height = viewbox_match.group(2)
    
    # Round to integers for width/height attributes
    width_int = int(float(width) * 0.8 + 0.5)
    height_int = int(float(height) * 0.8 + 0.5)
    
    # Remove width="100%"
    content = content.replace('width="100%"', '', 1)
    
    # Add width and height before the closing > of the svg tag
    # Find the end of the opening <svg ...> tag
    svg_open_pattern = re.compile(r'(<svg\s[^>]*?)(>)')
    
    def add_dimensions(match):
        return f'{match.group(1)} width="{width_int}" height="{height_int}">'
    
    content = svg_open_pattern.sub(add_dimensions, content, count=1)
    
    file_path.write_text(content, encoding='utf-8')
    return True


def process_mermaid_svgs(build_dir: str = "build/html/_images") -> None:
    """
    Process all mermaid-*.svg files in the specified directory.
    """
    images_path = Path(build_dir)
    
    if not images_path.exists():
        print(f"Directory not found: {images_path}")
        return
    
    svg_files = list(images_path.glob("mermaid-*.svg"))
    
    if not svg_files:
        print(f"No mermaid-*.svg files found in {images_path}")
        return
    
    print(f"Found {len(svg_files)} mermaid SVG file(s)")
    
    modified_count = 0
    for svg_file in svg_files:
        if fix_mermaid_svg(svg_file):
            print(f"  ✓ Modified: {svg_file.name}")
            modified_count += 1
        else:
            print(f"  - Skipped: {svg_file.name} (no changes needed)")
    
    print(f"\nModified {modified_count}/{len(svg_files)} files")


# For use as Sphinx extension
def on_build_finished(app, exception):
    """Sphinx event handler called when build finishes."""
    if exception is None:
        build_dir = Path(app.outdir) / "_images"
        process_mermaid_svgs(str(build_dir))


def setup(app):
    """Sphinx extension setup."""
    app.connect('build-finished', on_build_finished)
    return {'version': '1.0', 'parallel_read_safe': True}


if __name__ == "__main__":
    process_mermaid_svgs()