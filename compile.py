#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
import shutil
import sys

LATEX_COMPILE_MAX_ERROR_OFFSET = 256
VERBOSE_MODE = False

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
    parser.add_argument('--target', choices=['pdf', 'html'], help='Target format: pdf or html')
    parser.add_argument('--source-path', help='Path to the original document')
    parser.add_argument('--target-path', nargs='?', default=None, 
                        help='Target path (optional, default: ./dist/<name of the tex doc>)')
    parser.add_argument('--clean', action='store_true', help='Clean temporary files after compilation')
    parser.add_argument('--hyperref', action='store_true', help='Support hyperref package (only for pdf)')
    parser.add_argument('--verbose', action='store_true', help='Print debugging information')

    
    args = parser.parse_args()

    if args.hyperref and args.target != 'pdf':
        print("hyperref is only supported for pdf target")
        sys.exit(1)
    
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

    success, error_msg = compile_tex_to_pdf(
        args.source_path, 
        args.target_path, 
        "./temp", 
        clean_temp=args.clean,
        hyperref=args.hyperref
    )
    if not success:
        print(f"Error compiling document: \n\n{error_msg}\n\n")


if __name__ == '__main__':
    main()
    sys.exit(0)


