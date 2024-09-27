# Introduction to Vectors

Many simple measurements are in the form of **scalars**, or single numeric
values. Common examples include height and weight.

Some quantities, however, are more complicated. For example, when studying
air currents, wind has a particular *speed* and a *direction*. In this
case, we have multiple numbers which represent the measurement. These
values are referred to as **vectors** [^vector-despicable-me]. Vectors
group together multiple numbers to provide more complicated meanings.

## Vector Coordinate Systems

In 2D or 3D, vectors can be thought of as *directed quantities*,
represented as arrows. A 3D vector has 3 *components*, representing its
length along the *x*, *y* and *z*-axes.

![Diagram showing the Cartesian components of a sample vector in 3D.
](assets/vectors_intro/cartesian_coordinates.png){align=center}

These axes provide the **Cartesian coordinates** of points and vectors.

Vectors can also be represented using **spherical coordinates**, where the
**orientation** and **magnitude** are uncoupled. Orientation is represented
using two angles, $\phi$ and $\theta$, where $\phi$ measures the
inclination with respect to the positive *z*-axis and $\theta$ measures the
clockwise angle with respect to the positive *y*-axis. To avoid having two
representations for the same point, these angles are restricted to specific
ranges, with $0\leq\phi\leq180$ and $0\leq\theta<360$.

![Diagram showing the spherical components of a sample vector in 3D.
](assets/vectors_intro/spherical_coordinates.png){align=center}

The magnitude is simply calculated as the vector norm. Thus, the magnitude
of a 3D vector $\vec{v}$ is calculated as:
$$
    \| \vec{v}\| = \sqrt{v_x ^2 + v_y^2 + v_z^2}
$$
where $v_x,v_y,v_z$ are the respective *x, y, z* vector components,
described above.

## Vectors and Axes

There are two common types of orientated data: **vectorial** data and
**axial** data [^fisher-lewis-embleton].

* **Vectorial** data, represented as arrows, are pointed in a certain
  direction, which is distinct from the reverse direction.
* **Axial** data, represented as lines, are aligned with a certain
  orientation, where forward and reverse are considered identical.

![The difference between vectorial and axial data.
](assets/vectors_intro/axial_vector.png){align=center}

For example, an **escalator** moves only up or down. Typically, if you are
on the up escalator, you can only move up. You are restricted to a single
direction. A down escalator provides motion in the opposite direction. The
escalator represents *vectorial* data.

Meanwhile, a **staircase** does not have this restriction. Stairs do not
inherently move up or down [^fiddler-on-the-roof] and thus only have an
orientation but no direction. Stairs thus represent *axial* data.

## Examples of Vectors

Vector measurements are present in many different fields. Intuitive
examples include wind and water currents and motion in space. In
computational anatomy and material science, structural anisotropy can be
represented as an orientation field. At each position in a structure, the
computed axis is oriented in the dominant structural orientation, while the
magnitude reflects the degree of co-alignment of the local structures.

[^fisher-lewis-embleton]: Fisher, N. I., Lewis, T., & Embleton, B. J.
       J. (1993). Statistical analysis of spherical data ([New ed.], 1.
       paperback ed). Cambridge Univ. Press.

[^vector-despicable-me]: Need a clearer explanation? A well-known
[super-villain](https://youtu.be/A05n32Bl0aY?si=0br_2aCtqGcpMtkR) may be
able to help.

[^fiddler-on-the-roof]: Unless you are a certain
[milkman](https://youtu.be/W3Z-8U5mb7M?feature=shared) singing about his
dream home.