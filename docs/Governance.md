(governance-model)=
# Governance Model of DeepLabCut
(adapted from https://napari.org/stable/community/governance.html)

## Abstract

The purpose of this document is to formalize the governance process used by the
`DeepLabCut` project, to clarify how decisions are made and how the various
elements of our community interact.

This is a consensus-based community project. Anyone with an interest in the
project can join the community, contribute to the project design, and
participate in the decision making process. This document describes how that
participation takes place, how to find consensus, and how deadlocks are
resolved.

## Roles And Responsibilities

### The Community

The `DeepLabCut` community consists of anyone using or working with the project
in any way.

### Contributors

A community member can become a contributor by interacting directly with the
project in concrete ways, such as:

- proposing a change to the code via a GitHub
  [GitHub pull request](https://github.com/DeepLabCut/DeepLabCut/pulls);
- reporting issues on our
  [GitHub issues page](https://github.com/DeepLabCut/DeepLabCut/issues);
- proposing a change to the [documentation](https://deeplabcut.github.io/DeepLabCut/README.html) via a
  GitHub pull request;
- discussing the design of the `DeepLabCut` or its tutorials on in existing
  [issues](https://github.com/DeepLabCut/DeepLabCut/issues) and
  [pull requests](https://github.com/DeepLabCut/DeepLabCut/pulls);
- discussing examples or use cases on the
  [image.sc forum](https://forum.image.sc/tags/DeepLabCut) under the #deeplabcut tag; or
- reviewing [open pull requests](https://github.com/DeepLabCut/DeepLabCut/pulls)

among other possibilities. Any community member can become a contributor, and
all are encouraged to do so. By contributing to the project, community members
can directly help to shape its future.

Contributors are encouraged to read the [contributing guide](https://github.com/DeepLabCut/DeepLabCut/blob/main/CONTRIBUTING.md).

### Core developers

Core developers are community members that have demonstrated continued
commitment to the project through ongoing contributions. They
have shown they can be trusted to maintain `DeepLabCut` with care. Becoming a
core developer allows contributors to merge approved pull requests, cast votes
for and against merging a pull-request, and be involved in deciding major
changes to the API, and thereby more easily carry on with their project related
activities. Core developers appear as organization members on the `DeepLabCut`
[GitHub organization](https://github.com/orgs/DeepLabCut/people) and are on our
[@deeplabcut/core-developers](https://github.com/orgs/DeepLabCut/teams/core-developers) GitHub team. Core
developers are expected to review code contributions while adhering to the
core developer guide. New core developers can be nominated by any existing core developer,
and for details on that process see our core developer guide.

### Steering Council

The Steering Council (SC) members are core developers who have additional
responsibilities to ensure the smooth running of the project. SC members are
expected to participate in strategic planning, approve changes to the
governance model, and make decisions about funding granted to the project
itself. (Funding to community members is theirs to pursue and manage). The
purpose of the SC is to ensure smooth progress from the big-picture
perspective. Changes that impact the full project require analysis informed by
long experience with both the project and the larger ecosystem. When the core
developer community (including the SC members) fails to reach such a consensus
in a reasonable timeframe, the SC is the entity that resolves the issue.

Members of the steering council also have the "owner" role within the [DeepLabCut GitHub organization](https://github.com/DeepLabCut/)
and are ultimately responsible for managing the DeepLabCut GitHub account, the [@DeepLabCut](https://twitter.com/DeepLabCut)
twitter account, the [DeepLabCut website](http://www.DeepLabCut.org), and other similar DeepLabCut owned resources.

The current steering council of DeepLabCut consists of the original developers:

- [Mackenzie Mathis](https://github.com/mmathislab)
- [Alexander Mathis](https://github.com/alexemg)

The SC membership is revisited every January. SC members who do
not actively engage with the SC duties are expected to resign. New members are
added by nomination by a core developer. Nominees should have demonstrated
long-term, continued commitment to the project and its [mission and values](mission-and-values). A
nomination will result in discussion that cannot take more than a month and
then admission to the SC by consensus. During that time deadlocked votes of the SC will
be postponed until the new member has joined and another vote can be held.

The DeepLabCut steering council may be contacted at `admin@deeplabcut.org`.

## Decision Making Process

Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place on the [issue tracker](https://github.com/deeplabcut/deeplabcut/issues). Occasionally,
sensitive discussions may occur on a private core developer medium.

Decisions should be made in accordance with the [mission and values](mission-and-values)
of the DeepLabCut project.

DeepLabCut uses a “consensus seeking” process for making decisions. The group
tries to find a resolution that has no open objections among core developers.
Core developers are expected to distinguish between fundamental objections to a
proposal and minor perceived flaws that they can live with, and not hold up the
decision-making process for the latter.  If no option can be found without
objections, the decision is escalated to the SC, which will itself use
consensus seeking to come to a resolution. In the unlikely event that there is
still a deadlock, the proposal will move forward if it has the support of a
simple majority of the SC.

Decisions (in addition to adding core developers and SC membership as above)
are made according to the following rules:

- **Minor documentation changes**, such as typo fixes, or addition / correction of a
  sentence, require approval by a core developer *and* no disagreement or requested
  changes by a core developer on the issue or pull request page (lazy
  consensus). Core developers are expected to give “reasonable time” to others
  to give their opinion on the pull request if they’re not confident others
  would agree.

- **Code changes and major documentation changes** require agreement by *one*
  core developer *and* no disagreement or requested changes by a core developer
  on the issue or pull-request page (lazy consensus). For all changes of this type,
  core developers are expected to give “reasonable time” after approval and before
  merging for others to weigh in on the pull request in its final state.

- **Changes to the API principles** require a dedicated issue on our
  [issue tracker](https://github.com/DeepLabCut/DeepLabCut/issues) and follow the
  decision-making process outlined above.

- **Changes to this governance model or our mission, vision, and values**
  require a  dedicated issue on our [issue tracker](https://github.com/DeepLabCut/DeepLabCut/issues)
  and follow the decision-making process outlined above,
  *unless* there is unanimous agreement from core developers on the change in
  which case it can move forward faster.

If an objection is raised on a lazy consensus, the proposer can appeal to the
community and core developers and the change can be approved or rejected by
escalating to the SC.
