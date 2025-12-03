# ============================================================================
#                    ANN UTILITIES
# ============================================================================

module ANNUtils

export buildMulticlassANN, trainMulticlassANN, multiclassANNCrossValidation

using Flux
using Flux.Losses
using Statistics
using Random

include("preprocessing.jl")
using .PreprocessingUtils

include("metrics.jl")
using .MetricsUtils

function buildMulticlassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numClasses::Int=3;
                            transferFunctions::AbstractArray{<:Function,1}=fill(relu, length(topology)))
    """
    Build ANN with softmax output for multiclass classification
    """
    ann = Chain()
    input_size = numInputs
    
    for (i, num_neurons) in enumerate(topology)
        ann = Chain(ann..., Dense(input_size, num_neurons, transferFunctions[i]))
        input_size = num_neurons
    end
    
    ann = Chain(ann..., Dense(input_size, numClasses), softmax)
    
    return ann
end

function trainMulticlassANN(topology::AbstractArray{<:Int,1},
                           trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                           validationDataset::Union{Nothing, Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}}=nothing,
                           transferFunctions::AbstractArray{<:Function,1}=fill(relu, length(topology)),
                           maxEpochs::Int=1000, learningRate::Real=0.01,
                           maxEpochsVal::Int=20,
                           classWeights::AbstractArray{<:Real,1}=[1.0, 2.0, 3.0])
    """
    Train multiclass ANN with weighted cross-entropy loss
    """
    (trainInputs, trainTargets) = trainingDataset
    numClasses = size(trainTargets, 2)
    
    ann = buildMulticlassANN(size(trainInputs, 2), topology, numClasses; transferFunctions=transferFunctions)
    
    function weighted_crossentropy(ŷ, y)
        ŷ_safe = clamp.(ŷ, 1e-7, 1 - 1e-7)
        weights_reshaped = reshape(Float32.(classWeights), :, 1)
        sample_weights = sum(y .* weights_reshaped, dims=1)
        ce_per_sample = -sum(y .* log.(ŷ_safe), dims=1)
        weighted_ce = ce_per_sample .* sample_weights
        return mean(weighted_ce)
    end
    
    loss_fn(m, x, y) = weighted_crossentropy(m(x), y)
    opt_state = Flux.setup(Adam(learningRate), ann)
    
    bestAnn = deepcopy(ann)
    bestValLoss = Inf
    epochsSinceImprovement = 0
    
    for epoch in 1:maxEpochs
        Flux.train!(loss_fn, ann, [(trainInputs', trainTargets')], opt_state)
        
        if validationDataset !== nothing
            (valInputs, valTargets) = validationDataset
            curr_val_loss = loss_fn(ann, valInputs', valTargets')
            
            if curr_val_loss < bestValLoss
                bestValLoss = curr_val_loss
                bestAnn = deepcopy(ann)
                epochsSinceImprovement = 0
            else
                epochsSinceImprovement += 1
            end
            
            if epochsSinceImprovement >= maxEpochsVal
                return bestAnn
            end
        end
    end
    
    return validationDataset !== nothing ? bestAnn : ann
end

function multiclassANNCrossValidation(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Int,1}},
        crossValidationIndices::Array{Int64,1};
        maxEpochs::Int=1000, learningRate::Real=0.01,
        maxEpochsVal::Int=15,
        classWeights::AbstractArray{<:Real,1}=[1.0, 2.0, 3.0],
        verbose::Bool=true)
    """
    Cross-validation for ANN parameter selection
    """
    inputs, targets = dataset
    numFolds = maximum(crossValidationIndices)
    
    all_predictions = []
    all_targets = []
    
    if verbose
        println("\n  Testing topology: $topology")
    end
    
    for fold in 1:numFolds
        testMask = crossValidationIndices .== fold
        trainMask = .!testMask
        
        train_indices_all = findall(trainMask)
        n_val = floor(Int, length(train_indices_all) * 0.10)
        shuffle!(train_indices_all)
        
        val_idx = train_indices_all[1:n_val]
        actual_train_idx = train_indices_all[n_val+1:end]
        
        rawTrainInputs = inputs[actual_train_idx, :]
        rawTrainTargets = targets[actual_train_idx]
        rawValInputs = inputs[val_idx, :]
        rawValTargets = targets[val_idx]
        rawTestInputs = inputs[testMask, :]
        rawTestTargets = targets[testMask]
        
        trainTargetsOneHot = oneHotEncoding(rawTrainTargets, [0, 1, 2])
        valTargetsOneHot = oneHotEncoding(rawValTargets, [0, 1, 2])
        
        normParams = calculateMinMaxNormalizationParameters(rawTrainInputs)
        trainInputsNorm = normalizeMinMax(rawTrainInputs, normParams)
        valInputsNorm = normalizeMinMax(rawValInputs, normParams)
        testInputsNorm = normalizeMinMax(rawTestInputs, normParams)
        
        ann = trainMulticlassANN(topology,
                                (trainInputsNorm, Bool.(trainTargetsOneHot));
                                validationDataset=(valInputsNorm, Bool.(valTargetsOneHot)),
                                maxEpochs=maxEpochs,
                                learningRate=learningRate,
                                maxEpochsVal=maxEpochsVal,
                                classWeights=classWeights)
        
        testOutputs = ann(testInputsNorm')'
        testPredictions = [argmax(testOutputs[i, :]) - 1 for i in 1:size(testOutputs, 1)]
        
        push!(all_predictions, testPredictions)
        push!(all_targets, rawTestTargets)
    end
    
    all_preds_combined = vcat(all_predictions...)
    all_targets_combined = vcat(all_targets...)
    
    cm, acc, class_metrics, macro_f1, weighted_f1 = 
        confusionMatrixMulticlass(all_preds_combined, all_targets_combined, 3)
    
    if verbose
        println("    → Macro F1: $(round(macro_f1*100, digits=2))% | Weighted F1: $(round(weighted_f1*100, digits=2))%")
    end
    
    return macro_f1, weighted_f1, cm, class_metrics
end

end # module