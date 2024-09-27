# Statistics Overview

While histograms provide a useful means of visualising vectorial and axial
data in 3D, statistical tools provide **quantitative insights** into the
data.

```{caution}
You are likely familiar with *Euclidean statistics*, such as computing the
mean and variance of a set of values. Due to the different structure of
orientation vectors, we need to explore different statistical approaches.
```

To analyse and compare these types of data, we must use *directional
statistics*. [^fisher-lewis-embleton] [^mardia-jupp] Directional statistics
provide a means to study collections of oriented data in order to
understand dominant orientations and the spread of the data.

## Descriptive Statistics



## Vector Normalisation

Directional statistics consider pure directions and orientations, and thus
consider **unit vectors** (vectors with a magnitude of one). To be able to
apply these operations to non-unit vectors, normalisation must be
performed. *VectoRose* contains two approaches for normalisation:

1. Naïve normalisation.
2. Magnitude-weighted resampling.

### Naïve Normalisation

The simplest way to produce a set of unit vectors from a set of non-unit
vectors is to simply rescale each vector by the inverse of its magnitude.
This process produces a set with the same number of vectors, where all
vectors have a magnitude of 1 (except any zero-length vectors, which are
ignored). This rescaling removes effects of local vector magnitude, causing
all downstream analyses to only consider orientation.

### Magnitude-weighted Resampling

An alternative normalisation approach that does not eliminate magnitude
effects relies on magnitude-weighted resampling. As in the naïve case,
we begin by dividing each vector by its respective magnitude to yield a set
of pure orientations. We then resample these orientations using weights
proportional to the original vector magnitudes. Orientations (or
directions) corresponding to higher magnitudes are resampled more often,
reproducing the magnitude effects. The number of vectors to resample must
be selected in advance. The greater the number of vectors, the more similar
the new set of vectors will be to the original.


[^fisher-lewis-embleton]: Fisher, N. I., Lewis, T., & Embleton, B. J.
    J. (1993). *Statistical analysis of spherical data* ([New ed.], 1.
    paperback ed). Cambridge Univ. Press.

[^mardia-jupp]: Mardia, K. V., & Jupp, P. E. (2000). *Directional
    statistics*. J. Wiley.
