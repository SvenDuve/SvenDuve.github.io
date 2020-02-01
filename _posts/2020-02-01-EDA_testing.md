
# Title
> summary


```julia
using Distributions
using Plots
using Statistics
using DataFrames
using RDatasets
using StatsPlots
```

    ┌ Info: Precompiling Distributions [31c24e10-a181-5473-b8eb-7969acd0382f]
    └ @ Base loading.jl:1273
    ┌ Info: Precompiling Plots [91a5bcdd-55d7-5caf-9e0b-520d859cae80]
    └ @ Base loading.jl:1273
    ┌ Info: Precompiling DataFrames [a93c6f00-e57d-5684-b7b6-d8193f3e46c0]
    └ @ Base loading.jl:1273
    ┌ Info: Precompiling RDatasets [ce6b1742-4840-55fa-b093-852dadbb1d8b]
    └ @ Base loading.jl:1273
    ┌ Info: Precompiling StatsPlots [f3b207a7-027a-5e70-b257-86293d7955fd]
    └ @ Base loading.jl:1273


Lets have a look at famous some data.

```julia
gr()
iris = dataset("datasets", "iris")
first(iris, 5)
```




<table class="data-frame"><thead><tr><th></th><th>SepalLength</th><th>SepalWidth</th><th>PetalLength</th><th>PetalWidth</th><th>Species</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Categorical…</th></tr></thead><tbody><p>5 rows × 5 columns</p><tr><th>1</th><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>setosa</td></tr><tr><th>2</th><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td><td>setosa</td></tr><tr><th>3</th><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td><td>setosa</td></tr><tr><th>4</th><td>4.6</td><td>3.1</td><td>1.5</td><td>0.2</td><td>setosa</td></tr><tr><th>5</th><td>5.0</td><td>3.6</td><td>1.4</td><td>0.2</td><td>setosa</td></tr></tbody></table>



Simple scatter plot:

```julia
scatter(iris.PetalLength, iris.SepalLength, title="Petal vs. Sepal")
```




![svg](/images/EDA_testing_files/output_4_0.svg)



A quick look at the densities, can normality be assumed?

```julia
@df iris density(:PetalLength, group=:Species)
```




![svg](/images/EDA_testing_files/output_6_0.svg)



Lets extract some data to have a closer look at Setosa's Petal Length

```julia
setosaPetalData = iris[iris[!,:Species].=="setosa",:PetalLength]
```




    50-element Array{Float64,1}:
     1.4
     1.4
     1.3
     1.5
     1.4
     1.7
     1.4
     1.5
     1.4
     1.5
     1.5
     1.6
     1.4
     ⋮  
     1.3
     1.5
     1.3
     1.3
     1.3
     1.6
     1.9
     1.4
     1.6
     1.4
     1.5
     1.4



```julia
setosaPetalMean = mean(setosaPetalData)
setosaPetalVariance = var(setosaPetalData)
println("The sample mean: ", setosaPetalMean)
println("The sample variance: ", setosaPetalVariance)
```

    The sample mean: 1.462
    The sample variance: 0.030159183673469387


Lets see if we can fit a pdf to the data...

```julia
d = Normal(setosaPetalMean, sqrt(setosaPetalVariance))
histogram(setosaPetalData, bins=15)
plot!(twinx(), d, linecolor=:darkred, lw=3)
```




![svg](/images/EDA_testing_files/output_11_0.svg)



Although we have limited data, it seems the normality assumption holds. So this is it, a very simple EDA to see if I can succe
