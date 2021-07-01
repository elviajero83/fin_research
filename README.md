# Setup Your Dev Environment
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment) is the guide to the development environment setup. I provide a summary here:

1. install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) (miniconda is fine) 
2. from the Windows Start Menue, open `Anaconda Power Shell`
3. clone this repo and navigate to it
4. create a new env 

   `conda create --name aml pip==20.3.3 python=3.7`
   `conda activate aml`
5. install the requiremnts
   `pip install -r requirements.txt`

6. setup jupyter

   `conda install notebook ipykernel`
   `ipython kernel install --user --name aml --display-name "aml"`
   
7. Now you can start notebook and open one of the notebooks to start
   `jupyter notebook`
   open notebooks/08-Performance Evaluation
   