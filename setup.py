from setuptools import find_packages, setup
from typing import List

Hypen_E_Dot = '-e .'
def get_requirements(filepath:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements] 
        #while reading the requirements it also reads '\n'. this list comprehension removes that \n
        if Hypen_E_Dot in requirements:
            requirements.remove(Hypen_E_Dot)
            
        return requirements


setup(
    name="IMDB_SentimentAnalysis",
    version='0.0.1',
    author="Sjunaid",
    author_email="sjunaid@gmail.com",
    packages=find_packages(where="src"),
    install_requires = get_requirements('requirements.txt'),
    package_dir={"": "src"}
    
)


