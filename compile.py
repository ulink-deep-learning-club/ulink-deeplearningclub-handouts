#! /usr/bin/env python3

# TODO: assemble different headers for tex files for different purposes (pdf, html)

import argparse
import os
import subprocess
from pathlib import Path
import shutil
import sys
import re

LATEX_COMPILE_MAX_ERROR_OFFSET = 256
VERBOSE_MODE = False
NON_INTERACTIVE_MODE = False

def find_deepest_common_ancestor(paths: list[Path]) -> Path:
    """
    Find the deepest common ancestor directory of a list of paths.
    """
    if not paths:
        return Path.cwd()
    
    # Convert all paths to absolute paths and split into parts
    absolute_paths = [path.resolve().parts for path in paths]
    
    # Find the minimum length of path parts
    min_length = min(len(parts) for parts in absolute_paths)
    
    common_parts = []
    for i in range(min_length):
        # Check if all paths have the same part at position i
        current_part = absolute_paths[0][i]
        if all(parts[i] == current_part for parts in absolute_paths):
            common_parts.append(current_part)
        else:
            break
    
    return Path(*common_parts) if common_parts else Path('/')


def recursively_find_dependencies(tex_path: Path) -> set[Path]:
    """
    Recursively find all .tex files and media files included in the .tex files
    Searches `input`, `include`, and `includegraphics` commands in the .tex files
    """
    dependencies = set()
    visited = set()
    stack = [tex_path.resolve()]
    
    while stack:
        current_path = stack.pop()
        if current_path in visited:
            continue
        visited.add(current_path)
        
        try:
            content = current_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
            
        # Find \input and \include commands
        input_include_pattern = r'\\(?:input|include)\{([^}]+)\}'
        for match in re.finditer(input_include_pattern, content):
            included_file = match.group(1)
            included_path = current_path.parent / included_file
            
            # Try different extensions for .tex files
            if not included_path.suffix:
                included_path = included_path.with_suffix('.tex')
                
            if included_path.exists() and included_path not in visited:
                stack.append(included_path.resolve())
                dependencies.add(included_path.resolve())
                
        # Find \includegraphics commands
        graphics_pattern = r'\\(?:includegraphics|lstinputlisting)(?:\[[^\]]*\])?\{([^}]+)\}'
        for match in re.finditer(graphics_pattern, content):
            graphics_file = match.group(1)
            graphics_path = current_path.parent / graphics_file
            
            if graphics_path.exists():
                dependencies.add(graphics_path.resolve())
                
    return dependencies


def call_lwarpmk(cwd: Path, subcommands: list[str]):
    try:
        lwarpmk_html_cmd = [
            'lwarpmk',
            *subcommands
        ]
        
        result = subprocess.run(
            lwarpmk_html_cmd,
            cwd=str(cwd),  # Changed to temp project root
            capture_output=True,
            text=True
        )

        error_msg = ""
        if result.returncode != 0:
            error_msg = "\n".join(
                "===".join(
                    result.stdout.split("===")[1:-1]
                ).strip().splitlines()[:-1]
            )

        return result.returncode == 0, error_msg
    except Exception as e:
        return False, f"Python error: {str(e)}"

def compile_tex_to_html(project_root: Path, target_root: Path, temp_path: Path, clean_temp: bool = True):
    """
    Compile a TeX file to HTML using lwarpmk.
    
    Parameters:
    - project_root: Path to the project root directory
    - temp_path: Path for temporary files during compilation

    Returns:
    - tuple: (success: bool, parent_path: Path or None, error: str)
    """
    try:
        # Find the main .tex file in the project root
        tex_files = list(project_root.glob('*.tex'))
        if not tex_files:
            return False, "No .tex file found in project root"
        
        file_index = -1
        if "document.tex" in map(lambda x: x.name, tex_files):
            file_index = list(map(lambda x: x.name, tex_files)).index("document.tex")
        if len(tex_files) > 1 and (not NON_INTERACTIVE_MODE):  # If there are multiple .tex files, ask the user which one to compile
            file_index = -1
            print("Multiple .tex files found in project root. Which one to compile?")
            for i, tex_file in enumerate(tex_files):
                print(f"{i + 1}. {tex_file}")
            while file_index < 0 or file_index >= len(tex_files):
                file_index = int(input("Enter the number of the file to compile: ")) - 1
                if file_index < 0 or file_index >= len(tex_files):
                    print("Invalid input. Please enter a valid number.")
        
        main_tex = tex_files[file_index]
        
        # Find all dependencies
        dependencies = recursively_find_dependencies(main_tex)
        dependencies.add(main_tex.absolute())
        
        # Find the deepest common ancestor to maintain relative paths
        common_ancestor = find_deepest_common_ancestor(list(dependencies))
        
        # Create temp directory structure
        temp_path.mkdir(parents=True, exist_ok=True)
        target_root.mkdir(parents=True, exist_ok=True)

        if VERBOSE_MODE:
            print("Creating temp directory structure...")

        print(dependencies)
        
        # Copy all dependencies to temp directory maintaining relative structure
        for dep in dependencies:
            try:
                # Calculate relative path from common ancestor
                relative_path = dep.relative_to(common_ancestor)
                temp_file_path = temp_path / relative_path
                
                # Create parent directories if needed
                temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(dep, temp_file_path)
                if VERBOSE_MODE:
                    print(f"Copied: {dep} -> {temp_file_path}")
            except Exception as e:
                return False, f"Error copying file: {dep} Error: {str(e)}"

        
        if VERBOSE_MODE:
            print("Compiling TeX to HTML...")
        
        # Get the main tex file in temp directory
        temp_main_tex = temp_path / main_tex
        
        # Change to the temp directory (project root in temp)
        temp_project_root = (temp_path / project_root).resolve()
        
        # Step 1: Compile with XeLaTeX once
        if VERBOSE_MODE:
            print("Step 1.1: Compiling with XeLaTeX...")
        
        success, error = call_xelatex(temp_main_tex, temp_project_root)
        if not success:
            return False, f"XeLaTeX compilation 1 failed: {error}"

        if VERBOSE_MODE:
            print("Step 1.2: Compiling with XeLaTeX...")
        
        success, error = call_xelatex(temp_main_tex, temp_project_root)
        if not success:
            return False, f"XeLaTeX compilation 2 failed: {error}"

        
        # Step 2: Call lwarpmk html from the copied project directory
        success, error = call_lwarpmk(temp_project_root, ['html1'])
        if not success:
            return False, f"lwarpmk html1 second failed: {error}"
        
        # Step 3: Call lwarpmk limages from the copied project directory
        if VERBOSE_MODE:
            print("Step 3: Running lwarpmk limages...")
        
        success, error = call_lwarpmk(temp_project_root, ['limages'])
        if not success:
            print(f"WARN: lwarpmk limages failed: {error}")
        
        
        for file in temp_project_root.iterdir():  # iterate through the top level dictionary, copy html, css files.
            if (file.is_file()):
                file = file.resolve()
                if (
                    (file.suffix == '.html' and file.stem.endswith("html")) # check if the file is called *html.html
                    or (file.suffix == '.css')
                ): 
                    shutil.copy2(file, target_root)
                    if VERBOSE_MODE:
                        print(f"Copied: {file} -> {target_root}")
            elif (file.is_dir()):  # move the children directories
                shutil.copytree(file, target_root / file.name)
        html_files = list(target_root.glob('*.html'))
        if not html_files:
            return False, "No .html file found in target root"
        if (len(html_files) == 1):
            shutil.move(html_files[0], target_root / "index.html")

        # Clean up temp directory if needed
        if clean_temp:
            shutil.rmtree(temp_path)

        # Return the parent path of the temp directory (where output files are)
        return True, ""
        
    except Exception as e:
        return False, f"Error in call_lwarpmk: {str(e)}"

def call_xelatex(file_path: Path, temp_path: Path):
    """
    Compile a TeX file to PDF using XeLaTeX.
    
    Parameters:
    - file_path: Path to the input .tex file
    - temp_path: Path for temporary files during compilation

    Returns:
    - tuple: (success: bool, error: str)
    """

    xelatex_cmd = [
        'xelatex',
        '-interaction=nonstopmode',
        '-halt-on-error', # exit if an error occurs
        f'-output-directory={temp_path.resolve()}',
        str(file_path.resolve())
    ]
    if(VERBOSE_MODE): print(f"Calling XeLaTeX:\n{' '.join(xelatex_cmd)}\n")

    # Run xelatex from the temp directory to avoid permission issues
    result = subprocess.run(
        xelatex_cmd,
        cwd=str(file_path.parent.resolve()),  # Run from temp directory
        capture_output=True,
        text=True
    )

    error_msg = ""
    if result.returncode != 0 and result.stdout and "!" in result.stdout:
        start_index = result.stdout.index("!")
        offset = 0

        while (offset < LATEX_COMPILE_MAX_ERROR_OFFSET and (start_index - offset) > 0):
            if result.stdout[start_index - offset] == ")":
                break
            offset += 1

        error_msg = result.stdout[start_index - offset + 1:]

    return result.returncode == 0, error_msg.strip()

def compile_tex_to_pdf(tex_file_path: str, pdf_file_path: str, temp_file_path: str, clean_temp: bool = True, hyperref: bool = False):
    """
    Compile a TeX file to PDF using XeLaTeX.
    
    Parameters:
    - tex_file_path: Path to the input .tex file
    - pdf_file_path: Path where the output PDF should be saved
    - temp_file_path: Path for temporary files during compilation
    
    Returns:
    - tuple: (success: bool, output: str, error: str)
    """
    # Ensure paths are Path objects
    tex_path = Path(tex_file_path).resolve()
    pdf_path = Path(pdf_file_path).resolve()
    temp_path = Path(temp_file_path).resolve()

    # Create output directory if it doesn't exist
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory if it doesn't exist
    temp_path.mkdir(parents=True, exist_ok=True)
    
    # Clean temp directory
    for file in temp_path.iterdir():
        if file.is_file():
            file.unlink()

    # Compile TeX file to PDF
    for _ in range(1 if not hyperref else 2):
        success, error = call_xelatex(tex_path, temp_path)
        if not success:
            return False, error
        
    # Find the generated PDF file
    compiled_pdf_file = None
    for file in temp_path.iterdir():
        if file.suffix == '.pdf':
            compiled_pdf_file = file
            break
    
    if compiled_pdf_file and compiled_pdf_file.exists():
        # Move PDF to target location
        target_pdf = pdf_path if pdf_path.suffix == '.pdf' else pdf_path / f"{tex_path.stem}.pdf"

        Path(target_pdf.parent).mkdir(parents=True, exist_ok=True)  # Make sure the parent directory exists
        shutil.move(str(compiled_pdf_file), str(target_pdf))
        
        # Clean up temp files if requested
        if clean_temp:
            for file in temp_path.iterdir():
                if file.is_file():
                    file.unlink()
                    
        return True, ""

    return False, "No PDF file was generated"

def main():
    parser = argparse.ArgumentParser(description='Compile document to PDF or HTML')
    parser.add_argument('--target', choices=['pdf', 'html'], help='Target format: pdf or html', default='pdf')
    parser.add_argument('--source-path', help='Path to the original document')
    parser.add_argument('--target-path', nargs='?', default=None, 
                        help='Target path (optional, default: ./dist/<name of the tex doc>)')
    parser.add_argument('--clean', action='store_true', help='Clean temporary files after compilation', default=False)
    parser.add_argument('--hyperref', action='store_true', help='Support hyperref package (only for pdf)')
    parser.add_argument('--verbose', action='store_true', help='Print debugging information')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')


    args = parser.parse_args()

    if args.non_interactive:
        global NON_INTERACTIVE_MODE
        NON_INTERACTIVE_MODE = True

    if args.hyperref and args.target != 'pdf':
        print("hyperref is only supported for pdf target")
        sys.exit(1)

    if args.verbose:
        global VERBOSE_MODE
        VERBOSE_MODE = True
    
    if args.target_path is None:
        # Create default path if not provided
        base_name = ""
        source_path_obj = Path(args.source_path)
        base_name = source_path_obj.parent.name if source_path_obj.is_file() else source_path_obj.name
        args.target_path = os.path.join('./dist', base_name)
    
    if (VERBOSE_MODE):
        print(f"Target format: {args.target}")
        print(f"Original path: {args.source_path}")
        print(f"Target path: {args.target_path}")

    if args.target == 'pdf':
        success, error_msg = compile_tex_to_pdf(
            args.source_path, 
            args.target_path, 
            "./temp", 
            clean_temp=args.clean,
            hyperref=args.hyperref
        )
        if not success:
            print(f"Error compiling document: \n\n{error_msg}\n\n")
    elif args.target == 'html':
        success, error_msg = compile_tex_to_html(
            Path(args.source_path), 
            Path(args.target_path),
            Path("./temp")
        )
        if not success:
            print(f"Error compiling document: \n\n{error_msg}\n\n")


if __name__ == '__main__':
    main()
    sys.exit(0)


