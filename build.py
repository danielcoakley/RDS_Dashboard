import PyInstaller.__main__
import os
import shutil
import streamlit

def create_executable():
    # Get the streamlit package directory
    streamlit_dir = os.path.dirname(streamlit.__file__)
    
    # PyInstaller arguments
    args = [
        'app.py',  # Main script
        '--name=EnergyBaselineDashboard',  # Name of the executable
        '--onefile',  # Create a single executable file
        '--noconsole',  # Don't show console window
        '--add-data=seu_mapping.csv;.',  # Include the SEU mapping file
        f'--add-data={streamlit_dir};streamlit',  # Include streamlit package
        '--hidden-import=streamlit',
        '--hidden-import=streamlit.version',
        '--hidden-import=importlib.metadata',
        '--hidden-import=importlib_metadata',
        '--clean',  # Clean PyInstaller cache
        '--noconfirm',  # Replace existing spec file
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("Executable created successfully!")
    print("You can find it in the 'dist' folder.")

if __name__ == '__main__':
    create_executable() 