# Import Modules
```julia
using <library>

include("....jl")
```

# Define Function
```julia
# You can declare the types of function arguments by appending `::<type>` to the argument name.
# Keyword arguments can make these complex interfaces easier to use and extend by allowing arguments to be identified by name instead of only by position.
function <func>(<argument1>, <argument2>, ...; <argument3>, <argument4>, ...)
    # Returns the value of the last expression evaluated.
    [[return] ...]
end


# Assignment form
<func>(...) = ...


# Anonymous Function
x -> ...

# Example
x -> x^2 + 2x - 1
```

# If Statement
```julia
if ...
    ...
end
```

# `nothing`
- Python의 `None`과 유사합니다.

# Use Python in Julia
```julia
using PyCall

@pyimport <package>
# or
pyimport("<package>")
```

# Built-in Functions
## `println()`
## `filer()`
```julia
filter(<func>, <collection>)
```
## `map()`
```julia
filter(<funct>, <collection>)
```
## `length()`
## `repeat()`
```julia
# Examples
# `"GeeksforGeeksGeeksforGeeks"`
repeat("GeeksforGeeks", 2)

B = [1, 2, 3, 4]
# `[1 1; 2 2; 3 3; 4 4]`
repeat(B, 1, 2)
# `[1 1; 2 2; 3 3; 4 4; 1 1; 2 2; 3 3; 4 4]`
repeat(B, 2, 2)
```
## `get()`
```julia
get(<collection>, <key>, <default>)
# Example
get(d, "a", 3)
```
## Datatype(?)
```julia
typeof()
```
## Strings
### Convert to Strings
```julia
# Example
string.(<collection>)
```
### `rsplit()`
```julia
# Similar to `split`, but starting from the end of the string.
rsplit(<text>, <separator>, [limit])

# Example
# 파일 이름에서 마지막 확장자만 추출하는 함수입니다.
extension_λ(name) = rsplit(name, ".", limit=2)[end]
```
### Check If Substring
```julia
occursin("<substring>", "<string>)")
```
### Replace
```julia
replace("<text>", "<pattern1>" => "<pattern2>", [count])
```
## Arrays
```julia
# A 1D array can be created by simply writing array elements within square brackets separated by `,` or `;`. 
[<value1>, <value2>, ...]
Array([<value1>, <value2>, ...])
Array{<type>}([<value1>, <value2>, ...])

# A 2D array can be created by writing a set of elements without commas and then separating that set with another set of values by a `;`. 
[<value1> <value2> ... ; <value3>, <value4> ...]

# Similarly, a 3D array can be created by using `cat` command. This command takes arrays and number of dimensions required as input and then concatenates the arrays in given dimensions.
cat([<value1> <value2> ... ; <value3> <value4> ...], [<value5> <value6> ... ; <value7> <value8> ...], dims=3)
```
### Add Element
```julia
# At the end
push!(<Array>, <value>)

# At the beginning
pushfirst!(<Array>, <value>)

# At specific position
splice!(<Array>, <range>, <value>)

# Example
# `[1, 2, 3, 4, 5]` -> `[1, 2, "G", "F", "G", 5]`
splice!(Array1, 3:4, "GFG")
```
### Remove Element
```julia
# Delete last element
pop!(<Array>)

# Delete first element
popfirst!(<Array>)

# Deleting a range of elements
deleteat!(<Array>, <range>)

# Deleting all the elements
empty!(<Array>)
```
## Tuples
- Immutable.
```julia
(<value1>, <value2>, ...)


# Named Tuples
# Example
x = (a=1, b=2)
```
## Sets
```julia
Set([<value1>, <value2>, ...])
Set{<type>}([<value1>, <value2>, ...])
```
### Return Whether Element Is in Set
```julia
# 원소가 집합에 속하는지를 나타냅니다.
<value> ∈ <Set>
# 위와 같습니다.
in(<value>, <Set>)
```
### Add Element
```julia
# Julia allows adding new elements in a set with the use of `push!` command. unlike arrays, elements in set can’t be added on a specific index because sets are unordered.
push!(<Set>, <value>)
```
### Delete Element
```julia
delete!(<Set>, <value>)
```
### Merge Elements of Two Sets
```julia
union(<Set1>, <Set2>)
union!(<Set1>, <Set2>)
# the `union!` function will overwrite `<Set1>` with the new merged set. 
```
### `intersect()`, `setdiff()`
### `issetequal(<Set1>, <Set2>)`

# File System
```julia
pwd()
cd("<path>")
# 디렉토리에 있는 파일과 폴더를 출력합니다.
readdir("<path">)
# 폴더인지 확인합니다.
isdir("<path>")
# 파일인지 확인합니다.
isfile("<path>")
# 경로를 합성합니다.
joinpath("<path1>", "<path2>", ...)
```

# Index of Julia
- `0`이 아닌 `1`부터 시작됩니다.
- 마지막 Index를 나타내는 변수는 `-1`이 아니라 `end`입니다.

# Install Package
```julia
using Pkg

Pkg.add("<package")
```

# `DataFrames`
```julia
using DataFrames
```
## Create DataFrame
```julia
DataFrame(
    :col1 => <collection1>,
    :col2 => <collection2>,
    ...
)

DataFrame(
    col1 = <collection1>,
    col2 = <collection2>,
    ...
)
```
## Indexing
```julia
<df>[!, [<col1>, <col2>, ...]]
<df>.<col2>
```
## Values(?)
```julia
size(<df>)
nrow(<df>)
ncol(<df>)
```
## Insert Column
```julia
insertcols!(<df>, <position>, <col> => <collection>)
```
## Save DataFrame
```julia
using CSV
using XLSX

# As "csv"
CSV.write(file_p, df, delim=",", writeheader=is_add_header)
# As "xlsx"
XLSX.writetable()
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