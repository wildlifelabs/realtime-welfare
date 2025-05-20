# Adapted from https://github.com/foiegreis/Py2pdf/blob/main/py2pdf.py by Greta Russi (@foiegreis) 2024
# Requires having Ghostscript and Enscript installed and available on the path
# Works on Mac OSX (brew) and Linux (apt)
#
import subprocess
import argparse
import os


def gather(directory: str, suffix: list):
    files = []
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            my_suffix = os.path.splitext(path)[1].lower()
            if my_suffix.lower().strip() == suffix:
                files.append(path)
        elif os.path.isdir(path):
            if not path.endswith("venv"):
                files.extend(gather(path, suffix))
    return files


def process_file(input_filename, output_filename, typename="python"):
    # Create PostScript file using enscript
    eps_filename = f"{output_filename}/{os.path.basename(input_filename)}.eps"
    pdf_filename = f"{output_filename}/{os.path.basename(input_filename)}.pdf"
    enscript_command = [
        'enscript',
        f'--output={eps_filename}',  # Output file path
        f'--highlight={typename}',  # Specify Python syntax highlighting
        # '--no-header',  # No header option
        '--header=$n||Page $% of $=',
        f'--title={os.path.basename(input_filename)}',  # Sets title as the file name
        f'--font=Courier9',  # Font name and size
        f'--portrait',  # Orientation
        f'--color',  # Color coded, bool
        input_filename  # Input file path
    ]
    postscript_content = subprocess.check_output(enscript_command, universal_newlines=True)

    # Convert PostScript to PDF using ps2pdf
    ps2pdf_command = [
        'ps2pdf',  # Should be available if you have a TeX distribution installed
        f'{eps_filename}',
        f'{pdf_filename}'
    ]
    subprocess.run(ps2pdf_command, universal_newlines=True)
    try:
        os.remove(eps_filename)
    except FileNotFoundError:
        print(f"File {eps_filename} not found")

def main():
    # we map the file suffix search to an enscript language file syntax highlight support
    # See `enscript --help-highlight` for more information
    mapping = {
        ".py": "python",
        ".json": "javascript",
        "Dockerfile": "inf",
        ".txt": "inf",
        ".sh": "bash",
        "Makefile": "makefile"
    }
    parser = argparse.ArgumentParser(description='Process files')
    parser.add_argument('-i', '--input', help='Input Path', required=True)
    parser.add_argument('-o', '--output', help='Output Path', required=True)
    parser.add_argument('-s', '--suffix', help='File Suffix', required=True)
    parser.add_argument('--single', action='store_true')
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    file_suffix = args.suffix
    if not args.single:
        files = gather(input_path, file_suffix)
        os.makedirs(output_path, exist_ok=True)
        for file in files:
            process_file(file, output_path, mapping[file_suffix])
    else:
        process_file(input_path, output_path, mapping[file_suffix])


if __name__ == "__main__":
    main()
