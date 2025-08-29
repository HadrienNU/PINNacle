# Changelog – Modifications to PINNacle

This fork introduces several new features and changes compared to the original PINNacle repository.  
The following list documents all modifications and additions.

---

## 1. `benchmark_electromag.py`
- **New file**: A copy of the original benchmark script, modified to handle electromagnetic PDE systems and new plots (`vectors.png` for magnetic and `heatmaps.png` for charge density).

---

## 2. `baseclass.py`
- **Modified**: `create_model()` now supports two additional parameters:
  - `architecture`: specifies the model architecture (e.g., `"MLP"`, `"KAN"`, `"Deep Ritz"`).
  - `exp_name`: defines the experiment name, used by `model.py` to save activation pattern plots into the correct folder.

---

## 3. `src/pde/`
- **New files**:
  - `electromag.py`: defines two new electromagnetic PDE systems and their Deep Ritz variants as new classes (`Magnetism_2D`, `Magnetism_Ritz`, `Electric_2D`, `Electric_Ritz`).
  - `simple_test.py`: implements a simple Poisson test system as new classes (`KAN_Test` and `DeepRitz_Test`).
- **Modified**:
  - `poisson.py`: created `Poisson_Ritz` to include a Deep Ritz variation by subclassing `Poisson2D`.

---

## 4. `deepxde/geometry/geometry_2d.py`
- **Modified**: added support for elliptical domain shapes by adding the `Ellipse` class.

---

## 5. `deepxde/model.py`
- **Modified**:
  - Integrated support for the `architecture` parameter.
  - Implemented KAN grid update functionality.
  - Added plotting of activation patterns.

## 6. `Command Line`
Usual command used is:
```
python benchmark_electromag.py --name "<experiment_name>" --method <method> --iter <number_epochs> loss-weight "<loss1>, <loss2>, ..., <loss_n>"
```
Other possible argument flags are listed in the benchmark_electromag.py file. For the two electromag PDE systems, it is possible to change the spatial domain shape through the command line 'name' argument. If the experiment name contains the string 'disk', the domain will be circular, 'ellipse' is for elliptical domain and 'polygon' is for L shape domain. Below we present the command lines that generated the best results for each case.
- Magnetic MLP Adam (same for all domain shapes):
```
python benchmark_electromag.py --name "mag-adam-disk" --iter 60000
```
- Charge Density MLP Adam (same for all domain shapes):
```
python benchmark_electromag.py --name "elec-adam-ellipse" --iter 40000
```
- Poisson MLP Adam:
```
python benchmark_electromag.py --name "poisson-adam" --loss-weight "0.01, 1, 1"
```
- Charge Density KAN LBFGS (same for all domain shapes):
```
python benchmark_electromag.py --name "elec-kan-polygon" --method kan --iter 400
```
- Burgers KAN LBFGS:
```
python benchmark_electromag.py --name "burgers-kan" --method kan --iter 400
```
- Magnetic MLP Deep Ritz ReLU (same for all domain shapes):
```
python benchmark_electromag.py --name "mag-ritz-disk" --method deepritz
```
- Charge Density MLP Deep Ritz ReLU (same for all domain shapes):
```
python benchmark_electromag.py --name "elec-ritz-ellipse" --method deepritz
```
- Charge Density KAN Deep Ritz (same for all domain shapes):
```
python benchmark_electromag.py --name "elec-kan-ritz-ellipse" --method kan-deepritz
```
For the last case in specific, the model converges much faster, but we kept the number of epochs as default (20000) to get more visualizations of the activation patterns throughout training.
