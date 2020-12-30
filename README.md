# jet-regression

Jet regression operating on CMS Open Data jetNtuples. Uses PFCandidate level information to try and learn jet energy corrections.

## Run

`python main.py`

Note: Running the code for the first time will take significantly longer than in the subsequent runs. This is because the datasets are downloaded from the CERN Open Data portal and formatted. This process takes _O_(1 hour).