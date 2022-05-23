using Pkg

Pkg.add("Format")
Pkg.add("MD5")
Pkg.add("HTTP")
Pkg.add("JSON")
Pkg.add("ConfParser")
Pkg.add("CSV")
Pkg.add("DataStructures")
Pkg.add("XLSX")
Pkg.add("ArgParse")
Pkg.add("PyCall")

readdir("D:/Github/Data-Science-in-Python/Julia")

using Format

func(x) = rsplit(x, ".", limit=2)

text = "assd.asdas.asdas.heic"
func(text)

func(text)[end-1]

set = Set(["a", "b", "c"])

"a" ∈ set
in("a", set)

Set1 = Set(["Geeks", "For", "Geeks"])
println(Set1)

push!(set, "d")

set

function g(x,y)
    return x + y, x*y
end

g(3, 4)


function f(x,y)
    x + y, x*y
end

f(3, 4)

length(set)

repeat(["No heic"])

# Julia program to illustrate
# the use of Array repeat() method

# Constructing an array by repeating
# the specified 1D array with the
# specified number of times.
A = [1, 2, 3, 4];
println(repeat(A, 1))

# Constructing an array by repeating
# the specified 1D array with the
# specified number of times.
B = [1, 2, 3, 4];
println(repeat(B, 1, 2))
println(repeat(B, 2, 2))

# Constructing an array by repeating
# the specified 2D array with the
# specified number of times.
C = [1 2; 3 4];
println(repeat(C, 2))

# Constructing an array by repeating
# the specified 2D array with the
# specified number of times.
D = [1 2; 3 4];
println(repeat(D, 2, 2))

[1, 2, 3, 4]
Array([1, 2, 3, 4])
[1 2; 3 4]

string.([1 2;])

using DataFrames

using Pkg
Pkg.add("DataFrames")


df = DataFrame(
    :col1 => [1, 2, "d"],
    :col2 => ["a", "b", "c"]
)

df.col3 <= 1:size(df)[1]

df

function add_sid_df!(df; sid_col_n="sid")
	println("$(now_λ()) [INFO][df_utils.add_sid_df] Adding sid ...")

	if sid_col_n ∈ names(df)
		df[!, sid_col_n] = 1:nrow(df)
	else
		insertcols!(
			df,
			1,
			sid_col_n => 1:nrow(df)
		)
	end
end

names(df)

df[!, ["col1", "col2"]]
df.col1

ncol(df)

insertcols!(df, 2, "add" => [11, 12, 13])

insert!(df, 3, [11, 12, 13], :asdf)

import Pkg
Pkg.add("DataStructures")
Pkg.add("Format")
Pkg.add("MD5")
Pkg.add("HTTP")
Pkg.add("JSON")
Pkg.add("ConfParser")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("XLSX")
Pkg.add("PyCall")
Pkg.add("ArgParse")
Pkg.add("PyCall")
Pkg.add("rawpy")

pip install rawpy

using PyCall

@pyimport math
@pyimport pandas


ENV["PYTHON"]="/Users/jongbeom.kim/Desktop/myenv/venv/bin/python3.9"
Pkg.build("PyCall")