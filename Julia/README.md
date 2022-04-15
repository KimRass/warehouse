# Using in Jupyter Notebook
# Install Package
```julia
using Pkg
```
```julia
Pkg.add("...")
```
```python
typeof()
```
# `CSV`
```julia
using Pkg

Pkg.add("CSV")
Pkg.add("XLSX")
Pkg.add("DataFrames")
```
```julia
using CSV, XLSX, DatFrames
```
```julia
xf = XLSX.readxlsx("D:/부의.xlsx")
# XLSX.sheetnames(xf)
# sh = xf["Sheet1"]
# All data inside worksheet's dimension
# sh[:]
```
```julia
df = DataFrame(XLSX.readtable("D:/계정별명세서조회_20220128_원가명세서(최종).xlsx", "sheet1", infer_eltypes=true)...)
```
```julia
jm = XLSX.readdata("D:/계정별명세서조회_20220128_원가명세서(최종).xlsx", "sheet1!B1:E30")
```
```julia
XLSX.openxlsx(, [enable_cache=true])
# Cache is enabled by default, so if you read a worksheet cell twice it will use the cached value instead of reading from disk in the second time.
```
# `PyPlot`
```julia
Pkg.add("Plots")
Pkg.add("PyPlot")
using PyPlot, Plots
```
```julia
Plots.scatter([[markersize], [markercolor], [legend])
```