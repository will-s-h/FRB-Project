# Description of frbfuncs Folder

Warning: this file isn't rendered correctly on GitHub. View using VSCode with the Markdown Preview extension, or some other Markdown preview software with KaTeX support.

## TODO
- setup unittest in testfunctions.ipynb
- test comsology.py and ktau.py
- finish writing README.md files

## Cosmology

### Generic Constants
- $c = 299792458~\mathrm{m/s}$
- $\mathrm{km/s/Mpc} = 3.24077929 \times 10^{-20} \mathrm{~s}^{-1}$ 
- $\mathrm{Jy}\cdot\mathrm{ms} = 10^{-29}~\mathrm{J~m}^{-2}~\mathrm{Hz}^{-1}$
- $m_p = 1.6726219 \times 10^{-27} \mathrm{~kg}$
- $G = 6.67408 \times 10^{-11} \mathrm{~m}^3 \mathrm{~kg}^{-1} \mathrm{~s}^{-2}$
- $\mathrm{pc} \cdot \mathrm{cm}^{-3} = 3.08567758 \times 10^{22} \mathrm{~m}^{-2}$

### Cosmological Model Constants and Functions
- $\mathrm{cosm\_model} = [\Omega_M, \Omega_\Lambda, \Omega_b, H_0~(\mathrm{km/s/Mpc})] = [0.286, 0.714, 0.049, 70]$
- $H_0 = 70 \cdot 3.24077929 \times 10^{-20}~\mathrm{s}$
- $\mathrm{set\_default()}$: sets cosmological model to default values as shown here
- $\mathrm{update\_cmodel(new\_model)}$: updates $\mathrm{cosm\_model}$ to match $\mathrm{new\_model}$

### Helper Functions
- $\mathrm{sort\_by\_first}(\mathrm{list1},~\mathrm{list2})$: sorts $\mathrm{list1}$ and $\mathrm{list2}$ in increasing order of $\mathrm{list1}$. Sorts in place if values in $\mathrm{list1}$ are equal.
- $\mathrm{integrate\_mult\_ends}(\mathrm{func},~\mathrm{begin},~\mathrm{end})$: integrates $\mathrm{func}$ from $\mathrm{begin}$ to each value of $\mathrm{end}$.

### Distance Functions
- $E(z) = \sqrt{(1+z)^3 \Omega_M + \Omega_\Lambda}$
- $\displaystyle D_C(z) = \int \frac{dz}{E(z)}$ (comoving distance)
- $D_L(z) = (1+z)D_C(z)$ (luminosity distance)

### Dispersion Measures
- $\displaystyle C = \frac{3cH_0 \Omega_b}{8\pi G m_H}$
- $\displaystyle \left(\frac{dDM}{dz}\right)_\mathrm{Arcus} = C (X_\mathrm{H}+X_{\mathrm{He}})\frac{1+z}{E(z)}$
- $\displaystyle \left(\frac{dDM}{dz}\right)_\mathrm{Zhang} = C \left(\frac{7}{8} \cdot 0.84 \right) \frac{1+z}{E(z)}$