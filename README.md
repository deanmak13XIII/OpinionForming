# OpinionForming Software

## Preview
- This repository includes GIFs that show a sneak preview of how the OpinionFormingUI should appear and run. Feel free to check them out!

## Set-up Steps
### Cloning from Git
#### Installing Python:

1. Go to the Python download page: https://www.python.org/downloads/
2. Select the appropriate version for your operating system and click the download button.
Once the download is complete, run the installer and follow the installation wizard.

#### Installing Git:

1. Go to the Git download page: https://git-scm.com/downloads
2. Select the appropriate version for your operating system and click the download button.
3. Once the download is complete, run the installer and follow the installation wizard.

#### Cloning the Git repository:
1. Open your terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Type `git clone git@github.com:deanmak13XIII/OpinionForming.git` and press enter.
   1. If you prefer HTTPS, you can type `git clone https://github.com/deanmak13XIII/OpinionForming.git` and press enter.
4. This will download the repository to your local machine. 

#### Navigate to the cloned repository in your files:
1. In the terminal or command prompt, type `cd OpinionForming` and press enter. 
2. This will change your current directory to the cloned repository.

#### Install any necessary dependencies:
1. You can install any dependencies listed in the requirements.txt using pip, the Python package manager.
2. Type `pip install <package name>` and press enter.
3. Repeat this step for any other dependencies that need to be installed.

#### Run the application:

1. Once the dependencies are installed, you can run the application.
2. You can either run this script from an IDE like PyCharm, or you can run it from the command line by typing `python src\opinion_formingui.py` and pressing enter. 
3. This will start the application. The command line or Run tab should then start a web UI at the address: `http://127.0.0.1:8050/`


## How the graph data caching works
- When having cloned from git, the data folder contains only yaml files.
- The `sampled_matrices_v<>.yaml` files are where matrices created by the sample_matrices.py script are stored. These are randomly generated matrices and are not created the default way. They are saved and are the first source of cached data if no csv has been created using the OpinionFormingUI.
- The compressed csv files are however the primary source of cached data and contain all graph data including graph measures.
- If having created graphs with vertices count of up to n-vertices, all graphs up to n-vertices will be sourced from the csv and not the yaml files.
- If creating graphs of (n+1)-vertices, that extra 1 vertex graphs will be sourced from the yaml files as it will not exist in the csv yet. Saving the newly created graph data to csv will allow it to be retrieved the next time you run OpinionForminUI.

#### Running sample_matrices.py
1. To create/update the sample_matrices.yaml files, simply run `python src\sample_matrices.py <lower_bound_vertex_count> <upper_bound_vertex_count>` and press enter.
2. Please note that previous files will be replaced if their vertex count is specified.
