# TryOn-CM for the Fashion Compatibility Modeling

The tensorflow implemention for ACM SIGIR’2020 paper "Fashion Compatibility Modeling through a Multi-modal Try-on-guided Scheme" [[paper]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401047).

Learn more about the paper on [Paper Page](https://dxresearch.wixsite.com/tryon-cm).

# Dataset Preparison

- Download the FOTOS dataset on [Google Drive](https://drive.google.com/open?id=1-0wG_NXEEWMFe1JqOG2nGx3uQJDiVInS).

- Perform data_processing.py to prepare the required files: Image pixel array, image feature and vocabulary.

- To do: Expend the FOTOS with more outfit.

# Using the Pre-trained Models

- Download the trained models from [Page](https://drive.google.com/open?id=1nL4CuyEvafEx8hbpGVj0v81C1fETZjR0)-example.

- Put the checkpoints in *./checkpoint/*.
    
- Put the example testing set in */data_path/data/*.
    
- Edit the data_path in the line 382 of the file tryon-cm.py.
    
# Training anew

- Perform data_processing.py to prepare the Training, validation and testing set.

- Perform tryon-cm.py and the models will be saved in *./checkpoint/*.
