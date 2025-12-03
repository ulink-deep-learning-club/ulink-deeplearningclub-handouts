#!/usr/bin/env python3
"""
Auto-build all LaTeX documents to HTML and update the viewer's table of contents.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Configuration
VIEWER_DIST = Path("Viewer/dist/dist")
TOC_FILE = VIEWER_DIST / "toc.json"
LESSON_DIR_PATTERN = re.compile(r"^L\d+-.*")

def find_lesson_directories(root="."):
    """Return sorted list of lesson directories matching pattern."""
    root_path = Path(root)
    dirs = []
    for entry in root_path.iterdir():
        if entry.is_dir() and LESSON_DIR_PATTERN.match(entry.name):
            dirs.append(entry)
    return sorted(dirs, key=lambda d: d.name)

def get_main_tex_file(lesson_dir):
    """Return the main .tex file in the lesson directory."""
    tex_files = list(lesson_dir.glob("*.tex"))
    if not tex_files:
        return None
    # Prefer document.tex, otherwise the first .tex file
    for tex in tex_files:
        if tex.name == "document.tex":
            return tex
    return tex_files[0]

def extract_title_from_tex(tex_file):
    """Extract the title from the LaTeX file's \title{} command."""
    try:
        content = tex_file.read_text(encoding='utf-8')
        # Look for \title{...} (may have line breaks)
        # Simple regex that captures content within braces, ignoring nested braces for now
        match = re.search(r'\\title\{([^}]+)\}', content)
        if match:
            title = match.group(1).strip()
            # Remove formatting commands like \textbf{}
            title = re.sub(r'\\textbf\{([^}]+)\}', r'\1', title)
            title = re.sub(r'\\texttt\{([^}]+)\}', r'\1', title)
            return title.strip().lstrip("\\textbf{").rstrip("}")
    except Exception as e:
        print(f"Warning: Could not extract title from {tex_file}: {e}")
    # Fallback: generate title from directory name
    return tex_file.parent.name.replace("-", " ").replace("_", " ")

def compile_lesson_to_html(lesson_dir, target_dir):
    """Compile a lesson's LaTeX to HTML using compile.py."""
    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "compile.py",
        "--target", "html",
        "--source-path", str(lesson_dir),
        "--target-path", str(target_dir),
        "--non-interactive",
        "--verbose"
    ]
    print(f"Compiling {lesson_dir.name}...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: compilation failed")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    else:
        print(f"  Success")
        return True

def update_toc(lessons_info):
    """Update or create toc.json with the given lessons info."""
    # Load existing toc if exists
    if TOC_FILE.exists():
        with open(TOC_FILE, 'r', encoding='utf-8') as f:
            toc = json.load(f)
    else:
        toc = {"documents": []}
    
    # Create mapping from id to existing entry
    existing_ids = {doc["id"] for doc in toc["documents"]}
    
    # Add or update entries for each lesson
    for info in lessons_info:
        doc_id = info["id"]
        # Remove existing entry with same id
        toc["documents"] = [doc for doc in toc["documents"] if doc["id"] != doc_id]
        # Add new entry
        toc["documents"].append({
            "id": doc_id,
            "title": info["title"],
            "path": info["path"],
            "description": info.get("description", "")
        })
    
    # Sort documents by id (optional)
    toc["documents"].sort(key=lambda x: x["id"])
    
    # Write back
    with open(TOC_FILE, 'w', encoding='utf-8') as f:
        json.dump(toc, f, indent=2, ensure_ascii=False)
    print(f"Updated {TOC_FILE} with {len(lessons_info)} documents.")

def main():
    parser = argparse.ArgumentParser(description="Build all lesson documents to HTML and update TOC.")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation, only update TOC.")
    parser.add_argument("--skip-toc", action="store_true", help="Skip TOC update, only compile.")
    args = parser.parse_args()
    
    # Ensure viewer dist exists
    VIEWER_DIST.mkdir(parents=True, exist_ok=True)
    
    lesson_dirs = find_lesson_directories()
    if not lesson_dirs:
        print("No lesson directories found.")
        return
    
    lessons_info = []
    for lesson_dir in lesson_dirs:
        print(f"\nProcessing {lesson_dir.name}")
        main_tex = get_main_tex_file(lesson_dir)
        if not main_tex:
            print(f"  No .tex file found, skipping.")
            continue
        
        # Extract title
        title = extract_title_from_tex(main_tex)
        
        # Determine target directory
        target_dir = VIEWER_DIST / lesson_dir.name
        
        # Compile if not skipped
        if not args.skip_compile:
            success = compile_lesson_to_html(lesson_dir, target_dir)
            if not success:
                print(f"  Compilation failed, skipping TOC entry.")
                continue
        
        # Check if index.html exists
        index_html = target_dir / "index.html"
        if not index_html.exists():
            print(f"  Warning: {index_html} not found after compilation.")
        
        # Prepare TOC entry
        lessons_info.append({
            "id": lesson_dir.name.lower().replace("-", "_"),
            "title": title,
            "path": f"{lesson_dir.name}/index.html",
            "description": f"Compiled from {lesson_dir.name}"
        })
    
    # Update TOC if not skipped
    if not args.skip_toc and lessons_info:
        update_toc(lessons_info)
    
    print("\nDone.")

if __name__ == "__main__":
    main()