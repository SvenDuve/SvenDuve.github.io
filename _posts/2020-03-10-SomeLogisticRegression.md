
# Title
> summary


# Logistic regression on pneunomia radiological images

First we load the required libraries.

```julia
using Images
using ImageMagick
using FileIO
using Distributions
using Random
```

```julia
folders = ["/Users/svenduve/OneDrive/OU_Code/M347/chest_xray/chest_xray/test/NORMAL/",
            "/Users/svenduve/OneDrive/OU_Code/M347/chest_xray/chest_xray/train/NORMAL/",
            "/Users/svenduve/OneDrive/OU_Code/M347/chest_xray/chest_xray/val/NORMAL/",
            "/Users/svenduve/OneDrive/OU_Code/M347/chest_xray/chest_xray/test/PNEUMONIA/",
            "/Users/svenduve/OneDrive/OU_Code/M347/chest_xray/chest_xray/train/PNEUMONIA/",
            "/Users/svenduve/OneDrive/OU_Code/M347/chest_xray/chest_xray/val/PNEUMONIA/"];
```

Lets see how many files we have,

```julia
tagNormal = "NORMAL"
tagPneumonia = "PNEUMONIA"

fileList = []

for folder in folders
    for file in readdir(folder)
        push!(fileList, folder * "//" * file)
    end
end

println("The folders contain ", length(fileList), " Data.")
```

    The folders contain 5856 Data.


Lets define a function to reshap the pictures,

```julia
function fotoReshape(file)

    im = imresize(load(file), 224, 224)
    Float64.(im)

end

```




    fotoReshape (generic function with 1 method)



```julia
# empty array holding the arrays containting the greyscale
x = []
# empty array holding the category
y = []

```




    0-element Array{Any,1}



Now lets fill the x and y arrays, with the reshaped data and the respective label, meaning "Normal" or "Pneunomia"

```julia

for file in fileList

    try
#        resizeIm = imresize(load(file), 224, 224)
        push!(x, fotoReshape(file)) #Float64.(resizeIm)) #reshape(Float64.(resizeIm), 1, :))
    catch
        continue
    end


    if occursin(tagNormal, file)
        push!(y, 0)
    elseif occursin(tagPneumonia, file)
        push!(y, 1)
    else
        error("This file is undetermined")
    end

end

```

A random photo from the files:

```julia
randomPic = Gray.(x[211])
```




![png](/images/SomeLogisticRegression_files/output_12_0.png)



Lets zoom in to an area in the middle

```julia
randomPic[80:110, 80:110]
```




![png](/images/SomeLogisticRegression_files/output_14_0.png)



The numerical represantion of this snippet is as follows

```julia
Float32.(randomPic[80:110, 80:110])
```




    31×31 Array{Float32,2}:
     0.490196  0.447059  0.431373  0.392157  …  0.768627  0.752941  0.745098
     0.52549   0.454902  0.482353  0.45098      0.780392  0.768627  0.752941
     0.509804  0.494118  0.517647  0.466667     0.764706  0.741176  0.741176
     0.486275  0.501961  0.505882  0.443137     0.72549   0.709804  0.694118
     0.533333  0.54902   0.529412  0.454902     0.705882  0.721569  0.74902 
     0.513726  0.537255  0.537255  0.486275  …  0.760784  0.756863  0.756863
     0.513726  0.537255  0.509804  0.509804     0.788235  0.74902   0.760784
     0.447059  0.443137  0.415686  0.403922     0.756863  0.784314  0.764706
     0.282353  0.376471  0.396078  0.403922     0.745098  0.760784  0.760784
     0.270588  0.301961  0.392157  0.427451     0.760784  0.772549  0.752941
     0.25098   0.282353  0.411765  0.443137  …  0.74902   0.764706  0.745098
     0.32549   0.32549   0.454902  0.443137     0.760784  0.772549  0.768627
     0.392157  0.360784  0.466667  0.466667     0.756863  0.788235  0.776471
     ⋮                                       ⋱                      ⋮       
     0.52549   0.560784  0.482353  0.505882     0.8       0.784314  0.74902 
     0.611765  0.576471  0.509804  0.513726  …  0.737255  0.745098  0.713726
     0.615686  0.592157  0.537255  0.521569     0.72549   0.721569  0.729412
     0.572549  0.545098  0.556863  0.541176     0.705882  0.737255  0.737255
     0.552941  0.576471  0.533333  0.498039     0.737255  0.764706  0.756863
     0.580392  0.52549   0.482353  0.423529     0.780392  0.784314  0.788235
     0.572549  0.545098  0.564706  0.427451  …  0.792157  0.803922  0.772549
     0.635294  0.607843  0.533333  0.498039     0.788235  0.784314  0.768627
     0.556863  0.52549   0.490196  0.47451      0.8       0.784314  0.760784
     0.541176  0.556863  0.537255  0.509804     0.772549  0.776471  0.756863
     0.486275  0.47451   0.494118  0.470588     0.772549  0.776471  0.792157
     0.411765  0.380392  0.415686  0.415686  …  0.780392  0.776471  0.772549



Now the color shifts somewhere in the middle of the picture, lets see if we can somehow verify this simply with the number, a simple average on the left and the right side from the middle should do the trick,

```julia
left = mean(Float32.(randomPic[80:110, 80:95]));
right = mean(Float32.(randomPic[80:110, 96:110]));
println("The average gray on the left side is ", left)
println("The average gray on the right side is ", right)
println("The brightness ratio is ", right/left)
```

    The average gray on the left side is 0.4911292
    The average gray on the right side is 0.75389403
    The brightness ratio is 1.5350218


# The Regression

Now we have some x and y labeled data, lets do some more reshaping and formatting and then try to run a regression on them to separate healthy people from people suffering from pneumonia,

```julia
# Some data prepping
# reshaping the 224 by 224 matrices to line vectors
for i in 1:length(x)

       x[i] = reshape(x[i], 1, :)

end

# create a matrix from an Array of Arrays
x = vcat(x...)

# we want the features in the rows, so
x = transpose(x)
y = transpose(y)
```




    1×5573 LinearAlgebra.Transpose{Any,Array{Any,1}}:
     0  0  0  0  0  0  0  0  0  0  0  0  0  …  1  1  1  1  1  1  1  1  1  1  1  1



Lets set up some neccessary functions for the logistic regression, we require,

- A random split of our training and test data
- The sigmoid function denoted as $\sigma$
- An initialasation function for some random weights
- The forward propagation
- An optimisation function with some gradient descent
- A function that returns some metric how good the algorithm detects pneunomia
- Finally a function that embeds all the above function into one big model


```julia
# get a random vector for testing

function traintest(x, y, datasplit=0.8)

    # datasplit is 0.8/ 0.2 by default
    # returns x_train, x_test, y_train, y_test

    m,n = size(x)
    N = zeros(n)
    N[randsubseq(collect(1:size(x)[2]), 0.8)] .= 1

    return x[:,N .== 1], x[:,N .== 0], y[:,N .== 1], y[:,N .== 0]


end





function σ(z)

    1 ./ (1 .+ exp.(-z))

end





function initialiseWithZeros(dimData)

    w = zeros(dimData)
    b = 0

    return w, b

end


function propagate(w, b, x, y)


    m, n = size(x)
    A = σ(transpose(w) * x .+ b) # 1x3
    cost = -(1/n) * (y * transpose(log.(A)) + (1 .- y) * transpose(log.(1 .- A)))

    dw = (1/n) .* (x * transpose((A .- y)))
    db = (1/n) * sum(A .- y)

    grads = Dict("dw" => dw, "db" => db)

    return grads, cost


end







function optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=false)


    costs = []

    for i in 1:num_iterations

        grad, cost = propagate(w, b, X, Y)

        global dw = grad["dw"]
        global db = grad["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0
            push!(costs, cost)
        end

        # could add print cost function

    end

    params = Dict("w" => w, "b" => b)
    grads = Dict("dw" => dw, "db" => db)


    return params, grads, costs

end



function predict(w, b, x)

    m, n = size(x)
    y_prediction = zeros(1, n)


    A = σ(transpose(w) * x .+ b)

    for i in 1:size(A)[2]

        if A[i] > 0.5
            y_prediction[i] = 1
        end

    end

    return y_prediction

end



function model(x, y, num_iterations, learning_rate = 0.5, print_cost=false)


    x_train, x_test, y_train, y_test = traintest(x, y)
    w, b = initialiseWithZeros(size(x_train)[1])

    parameters, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate)

    w = parameters["w"]
    b = parameters["b"]


    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, x_train)


    println("Train accuracy: ", (sum(y_prediction_train .== y_train) / size(y_train)[2])*100)
    println("Test accuracy: ", (sum(y_prediction_test .== y_test) / size(y_test)[2])*100)

    d = Dict("costs"=>costs, "y_prediction_test" => y_prediction_test, "y_prediction_train" => y_prediction_train,
            "w"=>w, "b"=>b, "learning_rate"=>learning_rate, "num_iterations"=>num_iterations)


    return d


end


```




    model (generic function with 4 methods)



```julia
radModel = model(x, y, 100, 0.2)
```

    Train accuracy: 88.64042933810376
    Test accuracy: 89.19164396003633





    Dict{String,Any} with 7 entries:
      "w"                  => [0.0100991; 0.0650345; … ; 0.127471; 0.132542]
      "y_prediction_train" => [0.0 0.0 … 1.0 1.0]
      "y_prediction_test"  => [0.0 0.0 … 1.0 1.0]
      "b"                  => 0.018112
      "learning_rate"      => 0.2
      "num_iterations"     => 100
      "costs"              => Any[[NaN]]



In this case the model predicts with an accuracy of 89%
