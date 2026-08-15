# User guide

## Installation

This HMM-based pipeline reconstructs text that has been hashed using the pipeline available on: [https://github.com/CompNet/novelshare](https://github.com/CompNet/novelshare/tree/acl2026). To test the hashing pipeline, clone the novelshare repository and then follow the instructions in the library user guide to set up the novelshare environment. You can either hash your own data or hash the corpora in the data folder in the novelshare repository. 

## Experimentation

The HMM-based pipeline uses the corpora in the data folder in the novelshare repository as default. To test its efficiency on your own hashed text, replace ```"data/Moby_Dick/"``` and ```"data/Pride_and_Prejudice/"``` filenames in the ```if __name__ == "__main__"``` function at the end of the HMM_and_Viterbi.py file.
