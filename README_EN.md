# DFT on a GPU

Japanese version: [README.md](README.md)

## Summary

This repository takes a SMILES string, builds a 3D structure, and runs geometry optimization and TD-DFT as a batch. GPU acceleration makes the workflow practical for screening molecules that are too slow for a CPU-only calculation.

`gpu4pyscf` completes DFT for molecules of moderate size. There is still a limit set by GPU memory, but the molecules in the examples finish.

## Environment setup

Create the environment with `conda`, `mamba`, or `micromamba`.

```bash
# 1. Enter the cloned repository
cd /path/to/<your_repository_name>

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate it
conda activate gpu4pyscf
```

This installs PySCF, GPU4PySCF, MACE, cuTENSOR, and Jupyter. An NVIDIA driver that supports CUDA 12 is required separately. `environment.yml` points `pip` at the PyTorch CUDA 12.1 index so that the MACE stack (`torch==2.5.1+cu121`) can be installed.

## Input format

Put a CSV file in `data/` before starting a batch. It needs at least two columns.

- **`material` (or `index`)**: an ID or name, such as `BTD1`, `1`, or `OLED_Material_A`. The string is used as the output folder name and the log file name.
- **`SMILES`**: the structure, for example `C=1C=CC=CC1`.

**Example (`data/input_materials.csv`)**:

```csv
material,SMILES
benzene,C=1C=CC=CC1
BTD1,CC1(C)C2=CC(C3=C(N=S=N4)...
```

## Notebooks

### 1. `dft_sandbox.ipynb` (DFT optimization, batch)

GPU4PySCF optimizes each geometry and then runs the excited-state calculation.

- **Purpose**: screen HOMO, LUMO, S1, T1, and the oscillator strength *f* for many molecules.
- **How to run**: point the notebook at a CSV in `data/` and run the cells.
- **Output**: under `output/`, an optimized geometry (`_opt.xyz`), a Molden file, cube files, and a CSV of properties.

#### Example

The two molecules below are taken from an open-access paper that reports HOMO, LUMO, and S1.

Choi, M. G.; Lee, C. H.; Adachi, C.; Lee, S. Y. Highly Effective Thermally Activated Delayed Fluorescence Emitters Based on Symmetry and Asymmetry Nicotinonitrile Derivatives. *Molecules* **2022**, *27*, 8274. https://doi.org/10.3390/molecules27238274

**Example CSV (`data/test_dft.csv`)**:

```csv
material,SMILES
ICzCN,C=1C(N2C3=CC=CC=C3C4=...
ICzCYP,C1=2C=3C(=CC=CC3)N(C1=...
```

Set the input and output paths in the loading cell. Comments in the notebook are in Japanese.

```python
# 2. データセットの読み込み
csv_path = 'data/test_dft.csv'
# 結果を保存（および再開）するCSVファイルのパス
output_csv = 'output/test_dft_results.csv'

df = pd.read_csv(csv_path)
print(f"読み込まれた化合物数: {len(df)}")
df.head()
```

Leave the following cells as they are, and set the level of theory in the settings cell. These names match the paper: B3LYP/6-31G(d).

```python
# 5. Batch process the dataset
XC_FUNCTIONAL = 'b3lyp'
BASIS_SET = '6-31g*'
```

GPU4PySCF accepts the same functional and basis names as PySCF, including polarization and diffuse functions.

Running the batch cell writes the properties to the output CSV. The table below compares this B3LYP/6-31G(d) calculation with Figure 1 of the paper. Energies are in eV. The paper values are that figure's B3LYP/6-31G(d) results, not experimental measurements.

| Molecule | Source | HOMO | LUMO | S1 | T1 | f |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ICzCN | Paper, Fig. 1 | −5.02 | −2.16 | 2.50 | 2.49 | 0.0043 |
| ICzCN | This work | −5.07 | −2.15 | 2.58 | 2.56 | 0.0138 |
| ICzCYP | Paper, Fig. 1 | −4.97 | −2.16 | 2.48 | 2.46 | 0.0076 |
| ICzCYP | This work | −5.01 | −2.17 | 2.51 | 2.46 | 0.0074 |

ICzCYP is close. Its LUMO differs by about 0.01 eV. The larger gaps in the ICzCN oscillator strength come mainly from the acceptor torsion, with a smaller contribution from the resolution-of-the-identity (RI) approximation and the coarse integration grid (`grids.level = 1`).

### 2. `mace_tddft_sandbox.ipynb` (MACE optimization, then a DFT single point)

MACE-OFF23 optimizes the geometry quickly, and GPU4PySCF then evaluates the electronic structure on that geometry without a further DFT optimization. The force field reproduces ωB97M-D3(BJ)/def2-TZVPPD, so the geometry is not expected to match a B3LYP/6-31G(d) optimization.

- **Purpose**: a first pass over a larger set. Most of the time saved is in the geometry optimization.
- **How to run**: same pattern as `dft_sandbox.ipynb`.
- **Note**: molecules that contain an unsupported element, such as boron, are marked `fail` and can be sent through `dft_sandbox.ipynb` instead.

Both rows below use the same electronic-structure settings: B3LYP/6-31G(d), `grids.level = 1`, and the RI approximation for both Coulomb (J) and exchange (K). The geometry optimizer differs: geomeTRIC with DFT forces, or MACE-OFF23. Energies are in eV. The wall time covers optimization through TD-DFT. The table is the `level = 1` run. The default in `dft_sandbox.ipynb` is now `level = 3`, the PySCF default.

| Molecule | Optimization | HOMO | LUMO | S1 | T1 | f | Wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ICzCN | DFT | −5.07 | −2.15 | 2.58 | 2.56 | 0.0138 | 7 h 23 min |
| ICzCN | MACE | −5.08 | −1.92 | 2.81 | 2.80 | 0.0037 | 2 h 55 min |
| ICzCYP | DFT | −5.01 | −2.17 | 2.51 | 2.46 | 0.0074 | 7 h 32 min |
| ICzCYP | MACE | −4.96 | −2.04 | 2.59 | 2.59 | 0.0017 | 3 h 4 min |

HOMO stays within 0.05 eV. LUMO, S1, and *f* do not. MACE-OFF23 small, medium, and large differ in model size only. All three are trained at ωB97M-D3(BJ)/def2-TZVPPD, which includes dispersion and uses a much larger basis than B3LYP/6-31G(d) without dispersion. Switching to the large model does not remove this difference.

On these two molecules the phenyl–pyridine interplanar angles at the acceptor opened as follows.

| Molecule | DFT optimization | MACE-OFF23 |
| --- | --- | --- |
| ICzCN | 46.9°, 49.5° | 84.8°, 67.4° |
| ICzCYP | 33.4°, 16.8° | 54.9°, 21.2° |

The heavy-atom RMSD is 1.54 Å for ICzCN and 1.10 Å for ICzCYP. A B3LYP/6-31G(d) single point on the MACE geometry lies 9.2 kcal/mol (ICzCN) and 8.4 kcal/mol (ICzCYP) above the DFT-optimized structure. HOMO is localized on the donor and moves little. LUMO, S1, and the weak charge-transfer oscillator strength follow the acceptor torsion. Δ*E*ST of ICzCYP shrinks from 0.047 eV to 0.004 eV.

The MACE wall time is about one third of the DFT wall time, which is useful for a broad screen. To recover the B3LYP/6-31G(d) geometry, *f*, and Δ*E*ST, use the MACE geometry only as the starting point of a later DFT optimization.

### 3. `dft_resp_single.ipynb` (RESP charges, one molecule)

One molecule is optimized and evaluated with a basis that includes diffuse functions.

- **Purpose**: an electronic structure for RESP charges in Multiwfn, and an electrostatic-potential map. TD-DFT is not run.
- **How to run**: paste the SMILES into the `SMILES` variable.
- **Output**: `_opt.xyz`, a checkpoint, a Molden orbital file, and cube files for the ESP.

### 4. `dft_ip_reorg.ipynb` (hole-transport properties, one molecule)

The notebook optimizes the neutral molecule and the cation, then combines the single-point energies.

- **Purpose**: the hole reorganization energy (λh), the vertical ionization potential (vIP), and the adiabatic ionization potential (aIP), for use with Marcus theory.
- **How to run**: paste the SMILES into the notebook.
- **Output**: `result.txt` (vIP, aIP, λh) and the optimized geometries.

### 5. `dft_ea_reorg.ipynb` (electron-transport properties, one molecule)

The same procedure as the ionization notebook, using the neutral molecule and the anion.

- **Purpose**: the electron reorganization energy (λe), the vertical electron affinity (vEA), and the adiabatic electron affinity (aEA).
- **How to run**: paste the SMILES into the notebook.
- **Output**: `result.txt` (vEA, aEA, λe) and the optimized geometries.

## Directory layout

- **`data/`**
  Input only. Place the SMILES CSV files here.
- **`output/`**
  Results from the DFT notebooks and the single-molecule notebooks: `.log`, `.xyz`, `.cube`, `.molden`, `result.txt`, and the summary CSV.
- **`output_mace/`**
  Results from `mace_tddft_sandbox.ipynb`, including the summary CSV.

## Speed and memory (cuTENSOR, grid, and the memory pool)

`dft_sandbox.ipynb` uses `grids.level = 3`, the PySCF default, which is close to Gaussian `Integral=Fine`. The comparison table above was computed at the coarser `level = 1`. `mace_tddft_sandbox.ipynb` stays at `level = 1` for screening. `level = 3` uses more VRAM and time, so stop other CUDA jobs on an 8GB GPU.

`dft_sandbox.ipynb` returns the CuPy memory pool to the OS after the geometry optimization, after the singlet TD-DFT, and after each molecule. The ionization and electron-affinity notebooks do the same when they change charge state. cuTENSOR is preloaded with `ctypes`, and CuPy / GPU4PySCF then use it for tensor contractions.

## References

- **GPU4PySCF**:
  Li, R.; Sun, Q.; Zhang, X.; Chan, G. K.-L. Introducing GPU Acceleration into the Python-Based Simulations of Chemistry Framework. *J. Phys. Chem. A* **2025**, *129*, 1459–1468. https://doi.org/10.1021/acs.jpca.4c05876
- **MACE**:
  Batatia, I.; Kovács, D. P.; Simm, G. N. C.; Ortner, C.; Csányi, G. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. *Advances in Neural Information Processing Systems* **2022**, *35*, 11423–11436.
- **MACE-OFF23**:
  Kovács, D. P.; Moore, J. H.; Browning, N. J.; Batatia, I.; Horton, J. T.; Pu, Y.; Kapil, V.; Witt, W. C.; Magdău, I.-B.; Cole, D. J.; Csányi, G. MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules. *J. Am. Chem. Soc.* **2025**, *147*, 17598–17611. https://doi.org/10.1021/jacs.4c07099
- **PySCF**:
  Sun, Q.; Berkelbach, T. C.; Blunt, N. S.; Booth, G. H.; Guo, S.; Li, Z.; Liu, J.; McClain, J. D.; Sayfutyarova, E. R.; Sharma, S.; Wouters, S.; Chan, G. K.-L. PySCF: the Python-based simulations of chemistry framework. *WIREs Comput. Mol. Sci.* **2018**, *8*, e1340. https://doi.org/10.1002/wcms.1340
- **cuTENSOR**:
  NVIDIA Corporation. *cuTENSOR Library*.
- **CuPy**:
  Okuta, R.; Unno, Y.; Nishino, D.; Hido, S.; Loomis, C. CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations. *Proceedings of Workshop on Machine Learning Systems (LearningSys)* **2017**.
- **geomeTRIC**:
  Wang, L.-P.; Song, C. Geometry optimization made simple with translation and rotation coordinates. *J. Chem. Phys.* **2016**, *144*, 214108. https://doi.org/10.1063/1.4952956
- **Molecules paper used for comparison**:
  Choi, M. G.; Lee, C. H.; Adachi, C.; Lee, S. Y. Highly Effective Thermally Activated Delayed Fluorescence Emitters Based on Symmetry and Asymmetry Nicotinonitrile Derivatives. *Molecules* **2022**, *27*, 8274. https://doi.org/10.3390/molecules27238274
