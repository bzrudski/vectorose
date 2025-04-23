---
title: 'VectoRose: A new package for analysing and visualising 3D non-unit vectors in Python'
tags:
    - Python
    - anisotropy
    - directional statistics
    - trabecular bone
    - histograms
    - spherical histograms
authors:
  - name: Benjamin Z. Rudski
    orcid: 0009-0000-3423-0662
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Joseph Deering
    orcid: 0000-0003-2868-0944
    equal-contrib: true
    affiliation: 2
  - name: Natalie Reznikov
    orcid: 0000-0001-5293-4647
    affiliation: "1, 2, 3"
affiliations:
  - name: Department of Quantitative Life Sciences, McGill University, Canada
    index: 1
  - name: Faculty of Dental Medicine and Oral Health Sciences, McGill University, Canada
    index: 2
  - name: Department of Bioengineering, Faculty of Engineering, McGill University, Canada
    index: 3
date: 21 April 2025
bibliography: paper.bib
---

# Summary

In scientific analyses, 3D vector orientations are often as important as their magnitudes. For example, in meteorology, wind currents do not simply have a *speed*, but also a *direction* [@klinkComplementaryUseScalar1998]. Meanwhile, in 3D biological structures and assemblies, the (co-)orientation of structural units is a feature of interest that directly reflects structure-function relationships. Previous works have examined the orientation of myocardial fibres in the heart [@lombaertHumanAtlasCardiac2012b; @dileepCardiomyocyteOrientationRecovery2023], while others have focused on the coalignment, or *anisotropy*, of trabecular struts in cancellous bone [@reznikovTechnicalNoteMapping2022]. Whether analysing the vector orientations in isolation or in combination with their magnitudes, numerical representations alone are insufficient to provide a complete understanding of these datasets.

To gain such insight, we seek to visualise and quantitatively analyse sets of 3D non-unit vectors. Any intuitive visualisation should represent the distributions of vector magnitude and orientation, as well as the interplay between these two variables, thus enabling observers to identify patterns of dominant orientations and their associated magnitudes. Meanwhile, quantitative insights into these data require *directional statistics* to be applied instead of conventional Euclidean approaches [@fisherStatisticalAnalysisSpherical1993; @mardiaDirectionalStatistics2000]. To meet these needs, in this work, we present **VectoRose**, a new open-source Python package. To facilitate data visualisation of collections of non-unit vectors, VectoRose computes linear vector magnitude histograms, 3D spherical orientation histograms, and a novel type of nested 3D spherical histograms -- simultaneously capturing orientation and magnitude data. Our package also implements existing directional statistics approaches to enable statistical analysis of such vectorial data. VectoRose is agnostic to the workflow used to compute the datasets and can be applied in various scientific fields.

# Statement of need

@fisherStatisticalAnalysisSpherical1993 presented a foundation for directional statistics, providing numerous statistical tests and operations for analysing orientations of unit vectors in 3D. However, their seminal work did not offer a software implementation or modern workflows for data visualisation. In addition, their work is limited to pure directions and orientations (represented by unit vectors), preventing the analysis of non-unit vectors. While @fisherStatisticalAnalysisSpherical1993 discussed data visualisation, limited computing capabilities at the time prevented rich 3D plotting operations. Our work seeks to implement and extend the work by @fisherStatisticalAnalysisSpherical1993 to fit into the modern 3D quantitative analysis landscape. Through VectoRose, we provide users with functions to compute directional statistics and construct 3D orientation histograms using the Python programming language. Our package also extends the previous work by integrating analysis of non-unit vectors. We consider these vectors as a bivariate distribution, with magnitude and direction as linked but separate variables (\autoref{fig:stat-support}). The marginal distributions of each variable can be easily visualised using histograms; the 1D magnitude histogram is generated using **Matplotlib** [@hunterMatplotlib2DGraphics2007], while the direction histogram is presented on the surface of a sphere using **PyVista** [@banesullivanPyVista3DPlotting2019]. To visualise the complete bivariate distributions, showing the joint distribution of magnitude and direction, VectoRose uses a novel approach to generate a series of *nested* spherical histograms.

![Scalar data, consisting of real numbers, can be visualised using a 1D scalar histogram, while unit vectors, or *pure directions* and *pure orientations* can be visualised using spherical histograms. **VectoRose** introduces a novel type of histogram to visualise non-unit vectors, consisting of *nested spherical histograms*, that couples these two metrics.\label{fig:stat-support}](./figures/stat-support/stat-support.png)

Other works have previously implemented similar directional analysis approaches. @hacihabibogluSphstatPython32022 developed `sphstat`, a Python package implementing many of the methods described by @fisherStatisticalAnalysisSpherical1993. While `sphstat` provides many statistical approaches, it can only produce 2D scatter plots, not histograms. Three-dimensional histogram plotting is offered in the `spherical_stats` package developed by @schmitzSpherical_stats2021. However, that package only implements a limited subset of quantitative analyses. In addition, the 3D plotting method used relies on a congruent form of sphere patching, which results in discrepancies in face area [@beckersUniversalProjectionComputing2011]. This issue is mitigated in the `sphere-histo` software developed by the @3d-pliSpherehisto2022 group, which constructs spherical histograms using a triangulated sphere representation. Unfortunately, that software only generates histograms and does not offer any ability to integrate this plotting functionality into larger packages. Similarly, while OrientationPy developed by @vasileOrientationPy2022 offers better approaches for constructing 2D spherical plots using equal-area binning, as well as 3D spherical histograms, it also does not include any other statistical analyses. In addition, these packages provide few automated software tests, complicating bug detection. In contrast to these existing tools, VectoRose offers the ability to visualise and quantitatively analyse collections of unit and non-unit vectors in 3D, with intuitive Python functions, most of which are automatically tested.

# Research applications

We developed VectoRose in the context of scale-independent analysis of 3D anisotropy in porous, branching, fibrous, bicontinuous, or reticulated structures. In this context, vectors may have a magnitude in the interval between 0 and 1, and any orientation within the upper hemisphere. In \autoref{fig:bay-leaf}, we illustrate the data visualisation features of VectoRose using the structural anisotropy of a bay leaf. A collection of antiparallel vectors has been created to ensure the orientation histogram is plotted on both hemispheres (upper and lower). At the leaf midrib, many of the anisotropy vectors have a high magnitude and are oriented along the global $y$-axis. On either side of the midrib, the vectors exhibit low magnitudes and follow multiple orientations, representing higher-order veins. These two regimes are captured in the nested spherical histogram; the innermost shells, corresponding to low-magnitude vectors of the veins, mainly exhibit a *girdle* of vectors lying in the $xy$-plane (roughly, the plane of the leaf), while the outermost shells, corresponding to high-magnitude vectors, exhibit a single *cluster* of vectors aligned with the $y$-axis -- corresponding to the midrib. This and other biological examples will be showcased in an upcoming case study paper [@rudskiStatisticalAnalysesFeature2025].

![The bay leaf exhibits a high degree of anisotropy oriented along the midrib, while the higher-order veins display lower anisotropy values in various orientations. The diversity in vein orientations is reflected in a girdle distribution when considering all vectors. However, considering vectors at specific magnitude levels reveals orientation patterns corresponding to specific leaf structures. Figure adapted from [@rudskiStatisticalAnalysesFeature2025]. $n$ -- number of vectors; $\gamma$ -- Woodcock’s shape parameter [@woodcockSpecificationFabricShapes1977; @fisherStatisticalAnalysisSpherical1993]; $\zeta$ -- Woodcock’s strength parameter [@woodcockSpecificationFabricShapes1977; @fisherStatisticalAnalysisSpherical1993].\label{fig:bay-leaf}](./figures/bay-leaf/bay-leaf-figure-joss.png)

# Acknowledgements

BZR and NR are members of the Centre de recherche en biologie structurale \[Centre for Structural Biology Research\] funded by the Fonds de Recherche du Québec -- Santé (FRQS; grant 288558). NR is a member of the FRQS Network for Intersectorial Research in Sustainable Oral and Bone Health, and a McGill University William Dawson Scholar. NR acknowledges funding from the Natural Sciences and Engineering Research Council (NSERC) of Canada (grant RGPIN-2021-02658). JD is supported by a fellowship from NSERC. BZR is supported by scholarships from NSERC and FRQS. We thank the staff of the Integrated Quantitative Biology Initiative at McGill University, and Comet Technologies Canada for the free-of-charge academic licenses for Dragonfly 3D World. We are grateful to the students of the Reznikov Lab and to Nicolas Piché and Marc D. McKee for fruitful discussions, and to Lovéni Hanumunthadu for segmenting the bay leaf. We also thank the developers, maintainers and communities of the open source software used, including the Python programming language, as well as the NumPy, SciPy, PyVista, Matplotlib and Pandas packages.

# References
