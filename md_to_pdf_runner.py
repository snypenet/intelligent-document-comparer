import argparse
import markdown
import subprocess
import os

def convert_markdown_to_html(markdown_text):
    html = markdown.markdown(markdown_text)
    return html

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input markdown")
    parser.add_argument("--output", required=True, help="Path to output PDF")
    args = parser.parse_args()

    print(f"Converting {args.input} to {args.output}")

    with open(args.input, "r") as reader:
        markdown_string = reader.read()

    html_output = convert_markdown_to_html(markdown_string)
    
    # Create temporary HTML file
    temp_html = "temp_output.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_output)
    
    # Convert HTML to PDF using WeasyPrint
    weasyprint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weasyprint.exe")
    subprocess.run([weasyprint_path, temp_html, args.output])
    
    # Clean up temporary file
    os.remove(temp_html)
    
    print(f"Converted {args.input} to {args.output}")

if __name__ == "__main__":
    main()

