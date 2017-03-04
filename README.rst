====================
The EUGENE Project
====================

--------------------------------------------------------------------------------
A toolbox for detecting, comparing, and characterizing dynamical kinds
--------------------------------------------------------------------------------

Overview
====================

This software implements a variety of algorithms for assessing whether two or more systems belong to the same "dynamical kind." A system is a set of causally related variables. A dynamical kind is a class of systems that, while not identical in their causal structure, are sufficiently similar to support a rich set of generalizations across the class. 

The primary algorithm in this first release is a method for comparing the dynamical symmetries of two systems in order to determine whether they belong to the same dynamical kind. (This algorithm is implemented by methods in the compare.py module.) Note that this comparison is accomplished without any prior knowledge of the detailed dynamics of either system.

Theoretical Background
========================================
The algorithms for discovering and exploiting dynamical kinds that comprise the EUGENE project are motivated by a theory of dynamical kinds as "natural kinds" -- categories that support inductive learning. For a detailed account of this theory and its motivation, see Jantzen, B. (2015) “Projection, symmetry, and natural kinds,” *Synthese* **192** (11): 3617-3646. You can find the paper at the publisher's site `here <https://link.springer.com/article/10.1007%2Fs11229-014-0637-5>`_, or as a preprint from the author's `webpage <http://www.ratiocination.org/wp-content/uploads/2014/08/Jantzen__Projection_Symmetry_and_Natural_Kinds.pdf>`_.

Detection of dynamical kinds can be applied to a wide variety of problems in system identification, model validation, and automated scientific discovery. For an overview of some of these as well as a detailed characterization of the algorithms in this package, see https://arxiv.org/abs/1612.04933.

Usage
====================
The primary algorithm in this tool set is the ``CompareModels`` method in the ``compare`` module. (Please note that the ``categorize`` module is only a place holder -- the method it implements is both inefficient and inconsistent). The purpose of the ``CompareModels`` method is to decide whether symmetry curves sampled from two different systems are the same, and thus whether the two systems belong to the same dynamical kind. 

To decide whether two systems belong to the same dynamical kind using the ``CompareModels`` method, it is necessary to first acquire one or more samples of the symmetries characteristic of each system. These are stored in objects of the class ``SymModel``. The most important attribute of these objects is the ``_sampled_data`` attribute. It consists of a list of *m* representations of a symmetry -- generally a vector function from the set of "target variables" to itself. Each element of ``_sampled_data`` is thus a list of length *v*, where *v* is number of target variables. For each of these lists, the elements are numpy nd-arrays with shape (*p*, *v* + 1), where *p* is the number of points sampled. The first *v* columns give the value of all target variables in the untransformed system, while the last gives the value of one of the target variables to which that state is mapped in the transformed system.

``SymModel`` objects can be built directly, but to build them automatically from data acquired by sampling virtual physical systems, this is the procedure to follow:

1. For each system, build sensor and actuator objects (both are provided in the ``connect`` module for a variety of virtual systems, which are in turn to be found in the ``virtual_sys`` module).

2. For each system, assemble the sensors and actuators into an ``interface`` object.

3. Pass each interface object to the ``TimeSampleData`` method in order to acquire time-series data that is stored in ``DataFrame`` objects.

4. Initialize ``SymModel`` objects with the resulting data frames (the conversion to symmetry models takes place during initialization).

5. Finally, pass the pair of ``SymModel`` objects to the ``CompareModels`` method. The method returns 1 if the systems are of different dynamical kinds (if the symmetry models are different) and 0 otherwise.
 
The best way to gain familiarity with the procedure is to examine the demonstration methods discussed in the next section.

Demos
--------------------
We have packaged a demonstration module, LotkaVolterraDemo, with the core algorithms. The demo builds a couple of virtual systems whose dynamics are described by the competitive Lotka-Volterra equations for two species. The demo builds corresponding sensors and actuators for probing these systems, then collects data and invokes the core comparison algorithm to determine whether or not the systems belong to the same dynamical kind.

Contributors
====================
The project PI and principal code author is `Dr. Benjamin C. Jantzen <mailto:bjantzen@vt.edu>`_. His homepage and research blog can be found at http://www.ratiocination.org.

The following people have contributed ideas or code to this, or earlier versions of the EUGENE Project software:

- Colin Shea-Blymyer
- Joseph Mehr
- JP Gazewood
- Jack Parker
- Alex Karvelis

The PI would also like to thank Teddy Seidenfeld for helpful suggestions regarding the comparison algorithm. While any errors or deficiencies in the algorithm are solely those of the EUGENE team, Teddy's suggestions greatly influenced the final design.

Funding and Support
====================
This material is based upon work supported by the National Science Foundation under Grant No. 1454190. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

The project maintainers would also like to thank the Philosophy Department of Virginia Tech for their material and intellectual support.

A Note on the Name
====================
You may be wondering why this is called the "EUGENE" project. There are two reasons:

(1) The name honors physicist Eugene Wigner, whose ideas about the role of symmetry in physical theory have played a central role in motivating the theory of dynamical kinds underlying the discovery algorithms implemented in this package. (Of course, that's not to say that we believe Wigner would have endorsed out approach -- we make no such claim.)

(2) The name "Eugene" is derived from the ancient Greek word *eugenes*, meaning "well-born". This notion is doubly significant for our project. First, the methods of automated discovery pursued here are consonant with a view of scientific methodology and general epistemology that takes seriously the origins of a hypothesis -- if a hypothesis originates via a sound method of generation (if it is "well-born") then it is well-justified. Second, this is the first release of software implementing the first generation of algorithms constructed for the EUGENE project. We hope it is *eugenes*.
