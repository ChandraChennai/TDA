# TDA 
For generating the persistence diagrams as well as the persistence images we followed the approach of this paper: J. Phys. Chem. C 2020, 124, 17, 9360–9368.

Data used for the code is given in Data.txt file.

To create the persistence images run the code: 
python TDA.py path-for-cif-folder

Concatenated Persistence Images of 0D, 1D and 2D generated are given in Final_3 folder.

To run the ResNet-18 code:
First run: python dataset.py
Then just for vector: python trainVec.py
To run with vector with images: python trainImg.py
