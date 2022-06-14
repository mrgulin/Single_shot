# What is LPFET?
LPFET or Local Potential Functional Embedding Theory is a quantum chemistry embedding scheme developed at the _Laboratoire de Chimie Quantique de Strasbourg_. Original LPFET algorithm has been developed for a homogeneous Hubbard model [\[1\]](https://www.mdpi.com/2079-3197/10/3/45). 

This Github repository is a result of my internship in the above mentioned lab supervised by the professor Emmanuel Fromager.  Its result is a modified LPFET algorithm that can handle:
+ Heterogeneous Hubbard molecules with added nearest neighbour interactions (or V-term) with Hamiltonian in second quantization:
$$\hat H = - \sum_{ij}t_{ij} \hat{c}^{\dagger}_{i\sigma}\hat{c}^{ }_{j\sigma}+\sum_i U_{i} \hat{n}_{i\uparrow}\hat{n}_{j\downarrow}+\sum_i \hat{n}_i v^{ext}_{i}+\sum_{\langle i,j \rangle} V_{ij} \hat{n}_i \hat{n}_j$$
+ *Ab initio* molecules

The algorithm in this repository enables user to do both single impurity embedding (only one site/orbital is embedded) and block embedding (multiple sites are embedded).
Also unlike the algorithm that is described in my thesis where electrons in the cluster do not feel electrons in the environment, there is also implementation where occupied environment orbitals contribute to the Hamiltonian (active space calculations).
