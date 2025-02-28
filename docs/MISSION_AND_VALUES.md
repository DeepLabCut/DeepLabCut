(mission-and-values)=
# Mission and Values of DeepLabCut

This document is meant to help guide decisions about the future of `DeepLabCut`, be it in terms of
whether to accept new functionality, changes to the styling of the code or graphical user interfaces (GUI),
or whether to take on new dependencies, when to break into other repos, among other things. It serves as a point of
reference for core developers actively working on the project, and an introduction for
newcomers who want to learn a little more about where the project is going and what the team's
values are. You can also learn more about how the project is managed by looking at our
[governance model](governance-model).

## Our founding principles

The founding DeepLabCut team came together around a shared vision for building the first open-source animal pose
estimation framework that is:

- user defined pose estimation - i.e. species or object agnostic.
- access to SOTA deep learning models that can be swiftly re-trained for customized applications
- fast (GPU-powered)
- scalable (project focused for ease of portability and sharability)


As the project has grown we've turned these original principles into the mission statement and set of values that we
described below.

## Our mission

DeepLabCut aims to be **the animal pose software package for Python** and to **provide access to deep learning-based
pose estimation for people to use in their daily work** without the need to be able to program in a deep learning
framework. We hope to accomplish this by:

- being **easy to use and install**. We are careful in taking on new dependencies, sometimes making them optional, and
aim support a fully (Python) packaged installation that works cross-platform.

- being **well-documented** with **comprehensive tutorials and examples**. All functions in our API have thorough
docstrings clarifying expected inputs and outputs, and we maintain a separate
[tutorials and information website](http://deeplabcut.org).

- providing **GUI access** to all critical functionality so DeepLabCut can be used by people without coding experience.

- being **interactive** and **highly performant** in order to support large data pipelines.

- providing a **consistent and stable API** to enable plugin developers to build on top of DeepLabCut without their
code constantly breaking and to enable advanced users to build out sophisticated Python workflows, if needed.

- **ensuring correctness**. We strive for complete test coverage of both the code and GUI, with all code reviewed by a
core developer before being included in the repository.

## Our values

- We are **inclusive**. We welcome newcomers who are making their first contribution and strive to grow our most
dedicated contributors into [core developers](https://github.com/orgs/DeepLabCut/teams/core-developers).
We have a [Code of Conduct](https://github.com/DeepLabCut/DeepLabCut/blob/main/CODE_OF_CONDUCT.md) to make DeepLabCut
a welcoming place for all.

- We are **community-engaged**. We respond to feature requests and proposals on our
- [issue tracker](https://github.com/DeepLabCut/DeepLabCut/issues).

- We serve **scientific applications** primarily, over “consumer or commercial” pose estimation tools. This often means
prioritizing core functionality support, and rejecting implementations of “flashy” features that have little
scientific value.

- We are **domain agnostic** within the sciences. Functionality that is highly specific to particular scientific
domains belongs in plugins, whereas functionality that cuts across many domains and is likely to be widely used belongs
inside DeepLabCut.

- We value **education and documentation**. All functions should have docstrings, preferably with examples, and major
functionality should be explained in our [tutorials](http://deeplabcut.org). Core developers can take an active role
in finishing documentation examples.


## Acknowledgements

We share a lot of our mission and values with [`napari`](https://napari.org/stable/community/mission_and_values.html)
and [`scikit-image`](https://scikit-image.org/docs/stable/about/values.html) and acknowledge the influence of their
mission and values statements on this document.
