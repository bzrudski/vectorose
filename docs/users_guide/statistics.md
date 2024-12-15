# Statistics Overview

While histograms provide a useful means of visualising vectorial and axial
data in 3D, statistical tools provide **quantitative insights** into the
data.

As we discussed in {doc}`Introduction to Vectors <vectors_intro>`, vectors
have a **magnitude** and a **direction** or **orientation**.

The *magnitude* is simply a scalar value and can be analysed using usual
approaches and existing statistics packages. You are likely familiar with
Euclidean statistics.

```{attention}
In some cases, special care should be taken when analysing the vector
magnitudes to ensure that the approaches used produce results that can be
interpreted. Just because you can perform an operation *numerically* does
not guarantee that the result is valid *conceptually*.
```

The **direction** is a more complicated quantity. For example, let's say we
want to find the average direction between a unit vector with
$(\phi_1, \theta_1) = (90, 358)$ and one with
$(\phi_2, \theta_2) = (90, 2)$ (all angles in degrees). If we try to
average the angles, we'll get:

\begin{align}
\overline{\phi} &= (90 + 90) / 2 = 180 / 2 = 90\\
\overline{\theta} &= (358 + 2) / 2 = 360 / 2 = 180
\end{align}

So, the average direction would be $(\phi, \theta) = (90, 180)$, right?
**Wrong!** Our supposed *average* vector is actually backwards! So, we
can't just do operations on the spherical coordinates. We need a different
framework for doing statistics on the directions. The answer is
**directional statistics**.

```{seealso}
Interested in learning about directional statistics in depth? Consult the
textbooks by {cite:t}`fisherStatisticalAnalysisSpherical1993, 
fisherStatisticalAnalysisCircular1995, mardiaDirectionalStatistics2000`.
```

## Directional Statistics

Text goes here.