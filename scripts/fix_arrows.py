import os
import re

def fix_unicode_arrows(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Replace unicode arrow with ASCII arrow in print statements
                    # This regex is a bit broad but should cover most cases
                    new_content = re.sub(r'print\(f"(.*?)→(.*?)"\)', r'print(f"\1->\2")', content)
                    # Also non-f-strings
                    new_content = re.sub(r'print\("(.*?)→(.*?)"\)', r'print("\1->\2")', new_content)
                    
                    if new_content != content:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"Fixed arrows in {path}")
                except Exception as e:
                    print(f"Could not process {path}: {e}")

if __name__ == "__main__":
    fix_unicode_arrows(".")
