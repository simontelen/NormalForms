# NormalForms
Julia software for solving systems of polynomial equations via normal form methods.

Brief description of the files: 

<b>TNFtools.jl</b> : Auxiliary functions and struct definitions for the routines in denseTNF.jl, projHNF.jl, polyhedralTNF.jl and toricHNF.jl.

<b>denseTNF.jl</b> : Routines for solving generic, total degree systems using truncated normal forms. Implementation of [Thesis, Algorithm 4.1] and variants. 

<b>projHNF.jl</b> : Routines for solving zero-dimensional systems in projective space using homogeneous normal forms. Implementation of [Thesis, Algorithm 4.2] and variants.

<b>polyhedralTNF.jl</b> : Routines for solving generic members of a polyhedral family of systems using truncated normal forms. Implementation of [Thesis, Algorithm 5.3] and variants.

<b>toricHNF.jl</b> : Routines for solving zero-dimensional systems in a complete toric variety. Implementation of [Thesis, Algorithm 5.6] and variants. 

<b>X_examples.jl</b> : Several examples illustrating the use of the routines in X.jl. Some examples are taken from the references listed below, in which more information can be found on the implemented algorithms.

[TMVB] S. Telen, B. Mourrain, and M. Van Barel. Solving polynomial systems via truncated normal forms. SIAM Journal on Matrix Analysis and Applications, 39(3):1421–1447, 2018.

[Thesis] S. Telen, Solving systems of polynomial equations. Ph. D. Thesis, KU Leuven, 2020.

[Tel] S. Telen, Numerical root finding via Cox Rings. Journal of Pure and Applied Algebra, 224(9), 2020. 

[BT] M. Bender, S. Telen, Toric eigenvalue methods for solving polynomial systems. To appear on the arXiv, 2020.

[MTVB] B. Mourrain, S. Telen, and M. Van Barel. Truncated normal forms for solving polynomial systems: Generalized and efficient algorithms. Journal of Symbolic Computation, https://doi.org/10.1016/j.jsc.2019.10.009, 2019.

[TVB] S. Telen and M. Van Barel. A stabilized normal form algorithm for generic systems of polynomial equations. Journal of Computational and Applied Mathematics, 342:119–132, 2018.

There is also one example taken from my master's thesis, which is available at https://www.scriptiebank.be/sites/default/files/thesis/2016-09/masterpaper.pdf
