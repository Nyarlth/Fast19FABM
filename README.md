#  FAST19FABM

The Zernike fitted Beams of FAST L-band 19 Feeds array.

## Model Storage

These Beam models are saved as **HDF5** file at **./data/FAST_19FA.h5**.

Each Beam model of a different feed is stored in different groups. 

In each group, 

  there are two datasets:
      the **Beam Model** and the **Fitting Coefficients**;
  
  and five attributions:  
      **FittingRadius**, **max fitting Zernike order**, **the Model Polarization** and two **Coordinate axis bins**.

## Model usage

The script **./script/Zbeam.py** dependants **zernike** python model and provide the basic usage of these Beam models.
1. eval the model beam from the **Coefficients.**
2. get the **virtual scan responses** under these Zernike fitting beam models.

Seen in **./script/example.ipynb**.
