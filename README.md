# Planetary Systems
Provided was a large number of simulated planetray systems (./data). 
Since it is expensive to generate these planetary systems we are interested in a fast way to generate new, realistic planetary system without having to go throught the whole simulation.

# The dataset
The data consists of two csv with a large number of simulated planets.
Each planet is described by 4 parameters: system_number,a,total_mass,r
The easy dataset consists of 400.000 planets within 24.000 planetary systems of up to 20 planets
The harder one of 12.000 planets within ? planetary systems of up to ? planets.


# General procedure
Total_mass, distance and radius of the planets stretch over multiple magnitudes. To get this more machine learning compatible we normalize the data by taking the logarithm and further normalize mean and std of each column.

Our approach uses a Encoder-Decoder strategy to embed the variable length planetary systems into fixed size system-vectors.
Later on we sample random with similar distribution to the embedded system-vectors and decode them to generate realistic planetary systems. 


# Getting started

* setup .venv "python -m venv .venv"
* activate venv unix ". .venv/scipts/activate" or windows "./.venv/scripts/activate"
* install current project as editable package "pip install -e ."
* run the project: python scripts/main_train_autoencoder_and_generate_new_systems.py

