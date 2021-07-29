# Collider Simulator
Collider simulator for TAU Physics Laboratory C Course

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