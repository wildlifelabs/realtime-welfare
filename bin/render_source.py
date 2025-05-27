# Adapted from https://github.com/foiegreis/Py2pdf/blob/main/py2pdf.py by Greta Russi (@foiegreis) 2024
# Requires having Ghostscript and Enscript installed and available on the path
# Works on Mac OSX (brew) and Linux (apt)
# Modified extensively to support rendering an entire codebase to a selected output format
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


def process_file(input_filename, output_filename, typename="python", target="pdf", flatten=False):
    # Create PostScript file using enscript
    base_filename = f"{output_filename}{"_" if flatten else "/"}{os.path.basename(input_filename).replace(".","_")}"
    enscript_command = [
        "enscript",
        f"--output={base_filename}.{"eps" if target=="pdf" else target}",  # Output file
        f"--highlight={typename}",  # Specify Python syntax highlighting
        # "--no-header",  # No header option
        "--header=$n||Page $% of $=",
        "--footer=-",
        f"--title={os.path.basename(input_filename)}",  # Sets title as the file name
        f"--font=Courier8",  # Font name and size
        f"--portrait",  # Orientation
        f"--color",  # Color coded, bool
        "--style=emacs",
        f"--language={"PostScript" if target=="pdf" else target }",
        input_filename  # Input file path
    ]
    # block until complete
    postscript_content = subprocess.check_output(enscript_command, universal_newlines=True)
    if target=="pdf":
        # Convert PostScript to PDF using ps2pdf
        ps2pdf_command = [
            'ps2pdf',  # Should be available if you have a TeX distribution installed
            f'{base_filename}.eps',
            f'{base_filename}.pdf'
        ]
        subprocess.run(ps2pdf_command, universal_newlines=True)
        try:
            os.remove(f"{base_filename}.eps")
        except FileNotFoundError:
            print(f"File {base_filename}.eps not found")


def main():
    # we map the file suffix search to an enscript language file syntax highlight support
    # See `enscript --help-highlight` for more information
    mapping = {
        ".py": "python",
        ".json": "bash",
        "Dockerfile": "makefile",
        ".txt": "bash",
        ".sh": "bash",
        "Makefile": "makefile"
    }
    parser = argparse.ArgumentParser(description='Process files')
    parser.add_argument('-i', '--input',
                        help='Input Path',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Output Path',
                        required=True)
    parser.add_argument('-s', '--suffix',
                        help='File Suffix',
                        required=True)
    parser.add_argument("--target", "-t",
                        type=str,
                        choices=["pdf", "rtf", "html"],
                        required=True,
                        help="Target output: pdf | rtf | html")
    parser.add_argument('--single',
                        action='store_true',
                        help="process a single file, not a whole directory")
    parser.add_argument('--flatten',
                        action='store_true',
                        help="use the output path as a prefix, so 'my_path/output' would render 'my_path/output_hello_c.rtf'")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    file_suffix = args.suffix
    if not args.flatten:
        os.makedirs(output_path, exist_ok=True)
    if not args.single:
        files = gather(input_path, file_suffix)
        for file in files:
            process_file(file, output_path, mapping[file_suffix], args.target, args.flatten)
    else:
        process_file(input_path, output_path, mapping[file_suffix], args.target, args.flatten)


if __name__ == "__main__":
    main()
