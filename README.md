# <img alt="ColliPy" src="/branding/logo_header/logo_header_cut.jpeg" height="150">

Particle Detector Simulation Manager

## List of available particles

* photon
* electron
* positron
* muon
* mu-plus
* neutrino
* pi-0
* pi-plus
* pi-minus
* k-long
* k-short
* k-plus
* k-minus
* neutron
* proton
* antiproton
* lambda
* antilambda
* sigma-plus
* sigma-minus
* sigma-0

## Important things to consider

### Randomness is (very) limited
There is no way to change the random seed at the remote software.
As a consequence, same injections specifications for distinct accelerators, will result the same results.
With that said, running this package concurrently is supported and encouraged for some cases.

Examples for cases when you can use threading:
* When injecting multiple different particles
* When injecting same particle with different momentum
* etc.

## Development workflow

### Standard acronyms to start a commit message:
* API: an (incompatible) API change
* BENCH: changes to the benchmark suite
* BLD: change related to building collipy
* BUG: bug fix
* DEP: deprecate something, or remove a deprecated object
* DEV: development tool or utility
* DOC: documentation
* ENH: enhancement
* MAINT: maintenance commit (refactoring, typos, etc.)
* REV: revert an earlier commit
* STY: style fix (whitespace, PEP8)
* TST: addition or modification of tests
* REL: related to releasing collipy
