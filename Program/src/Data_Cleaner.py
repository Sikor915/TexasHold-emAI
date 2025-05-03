import re

def clean_trailing_dots_in_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace patterns like 12.34. with 12.34
    cleaned_content = re.sub(r'(\d+\.\d+)\.', r'\1', content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    print(f"Cleaned file saved: {file_path}")

# Example usage
file_path = "D:\\RZECZY_NA_ZAJECIA\\BIAI\\repo\\TexasHold-emAI\\KaggleDataSet\\test.txt"
clean_trailing_dots_in_file(file_path)
