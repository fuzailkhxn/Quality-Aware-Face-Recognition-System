import os
import zipfile

def create_zip():
    zip_filename = 'quality-aware-face-recognition-final.zip'
    zipf = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
    root_dir = '.'
    
    excludes = ['.venv', '__pycache__', '.git', '.pytest_cache', zip_filename, '.streamlit']
    
    for root, dirs, files in os.walk(root_dir):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in excludes and not d.startswith('.')]
        
        for file in files:
            if file in excludes or file.endswith('.pyc') or file.endswith('.pyo'):
                continue
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, root_dir)
            zipf.write(file_path, arcname)
            
    zipf.close()
    print(f"Zip file created successfully: {os.path.abspath(zip_filename)}")

if __name__ == '__main__':
    create_zip()
