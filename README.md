# Discrete choice models

**Electronic companion of the paper:** *"A Comparative Empirical Study of Discrete Choice Models in Retail Operations"* by

- Gerardo Berbeglia: Melbourne Business School, University of Melbourne, g.berbeglia@mbs.edu
- Agusti­n Garassino: School of Sciences, Universidad de Buenos Aires, and School of Business, Universidad Torcuato di Tella, Buenos Aires, Argentina, agarassino@dc.uba.ar
- Gustavo Vulcano: School of Business, Universidad Torcuato di Tella, and CONICET, Buenos Aires, Argentina, gvulcano@utdt.edu

This code was tested on:

- Ubuntu Linux v18.04, v20.04 operating systems.
- Python v2.7, v3.6
   
It is designed with focus on its reusability and maintainability, taking advantage of several object oriented programming (OOP) features of Python. 
The implementation of the different algorithms is intended to be self-explanatory in terms of the names of the objects and functions used along the way.

The dependencies found in the file **requirements.txt** are necessary in order to successfully execute the code. They can be installed by running:

```
$ pip install -r requirements.txt
```

For more instructions just run:

```
$ python estimate.py
```

More examples on how to use the code also can be found on the **examples** folder.

Some of the papers and books used as sources to implement the algorithms for the different choice-based demand models are:

- *"Discrete choice methods with simulation"*, by Kenneth E. Train. (used for nested logit and the MNL which were  optimized via the Python library SciPy)
- *"An expectation-maximization algorithm to estimate the parameters of the Markov chain choice model"* by Serdar Simsek and Huseyin Topaloglu.
- *"A market discovery algorithm to estimate a general class of non-parametric choice model."* by Garrett van Ryzin and Gustavo Vulcano (used for the rank list)
- *"An expectation-maximization method to estimate a rank-based choice model of demand."* by Garrett van Ryzin and Gustavo Vulcano (used for the rank list)
- *"Analysis of a Generalized Linear Ordering Problem via Integer Programming"* by Isabel Mendez-Diaz, Gustavo Vulcano and Paula Zabala
- *"A Conditional Gradient Approach for Nonparametric Estimation of Mixtures of Choice Models"* by S Jagabathula, L Subramanian, A Venkataraman (used for the latent class MNL)
- *"The exponomial choice model: A new alternative for assortment and price optimization"* by A AlptekinoÄŸlu, JH Semple (optimized via the Python library  SciPy, which implements standard nonlinear techniques)
- *"Halton sequences for mixed logit"* by Kenneth E. Train (used for the mixed logit)
