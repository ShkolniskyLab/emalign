<h1>emalign</h1>

A algorithm for aligning rotation, reflection, and translation between volumes. 

Current version: 1.0.0

Project's homepage: https://github.com/ShkolniskyLab/emalign

Date: 03/2022

Please cite the following paper when using this package: 
XXX

<h2>Recommended Environments:</h2>
The package has been tested on Ubuntu 18.04 and Windows 10. It should probably work on other versions of Windows and Linux, but has not been tested on them yet. Similarly for macOS.

* Python 3.6.0+ is required.

* The package makes use of the pyfftw package, which in turn uses the FFTW library. Before installing emalign make sure you have the FFTW library installed on your system: http://www.fftw.org/fftw3_doc/Installation-and-Customization.html#Installation-and-Customization


<h2>Install emalign</h2>
<h3>Install emalign via pip:</h3>
We recommend installing emalign via pip:


    $ pip install emalign

<h3>Install emalign from source</h3>
The tarball of the source tree is available via pip download emalign. You can install emalign from the tarball:


    $ pip install emalign-x.x.x.tar.gz


You can also install the development version of emalign from a cloned Git repository:


    $ git clone https://github.com/ShkolniskyLab/emalign.git

    $ cd emalign

    $ pip install .

<h2>Uninstall emalign</h2>
Use pip to uninstall emalign:


    $ pip uninstall emalign

<h2>Upgrade emalign</h2>
Just use pip with -U option:


    $ pip install -U emalign

<h2>Getting started:</h2>
Please read the user manual for usage instructions, available at the homepage of the project on Github: https://github.com/ShkolniskyLab/emalign


<h2>Usage:</h2>
Generate test data via: 


    $ vol1, vol2 = gentestdata(emdID) #emdID is an integer describing a map file from EMDB. 

Run the alignment algorithm via:


    $ vol2aligned = emalign(vol1, vol2) 
    
Verify the result by calculating, for example, the correlation between vol1 and vol2aligned.
