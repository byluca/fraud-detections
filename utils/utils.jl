#ONEHOT
function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]));
    
    # Second defensive statement, check the number of classes
    numClasses = length(classes);
    @assert(numClasses>1)
    
    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;

#function oneHotEncoding(feature::AbstractArray{<:Any,1}) 

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

# Alternative more compact definition

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end;

#function normalizeMinMax

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
   minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    normalizeMinMax!(copy(dataset), normalizationParameters);
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
      normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end;

#function normalizeZeroMean

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Remove any atribute that do not have information
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset; 
end;   

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset));   
end;  

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    normalizeZeroMean!(copy(dataset), normalizationParameters);
end;                    

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));
end;

#function classifyOutputs

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
   numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;

#function accuracy

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    accuracy(outputs.>=threshold, targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;

#function buildClassANN

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;


#function holdOut

function holdOut(N::Int, P::Real)
    @assert N ≥ 1 "N must be ≥ 1"
    @assert 0 ≤ P ≤ 1 "P must be in [0, 1]"
    
    # Calculate number of test patterns
    n_test = round(Int, P * N)

    # Generate random permutation of indices from 1 to N
    perm   = randperm(N)              # always shuffled

    # Split indices into test and training
    testIdx  = perm[1:n_test]
    trainIdx = perm[n_test+1:end]
    return (trainIdx, testIdx)
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert N ≥ 1 "N must be ≥ 1"
    @assert 0 ≤ Pval ≤ 1 "Pval must be in [0, 1]"
    @assert 0 ≤ Ptest ≤ 1 "Ptest must be in [0, 1]"
    @assert Pval + Ptest ≤ 1 "Pval + Ptest must be ≤ 1"

    # 1) First split: remove the test set from the full index space 1:N
    trainValIdx, testIdx = holdOut(N, Ptest)

    # If everything went to test, training/validation are necessarily empty
    n_rem = length(trainValIdx)
    if n_rem == 0
        return (Int[], Int[], testIdx)
    end

    # 2) Second split on the *remaining* indices:
    #    we need the validation fraction relative to the remaining pool.
    #    Example: N=100, Ptest=0.2, Pval=0.1  → remaining=80, val_rel = 0.1 / 0.8 = 0.125
    Pval_rel = Pval / (1 - Ptest)

    # Use the 2-way holdOut again on the remaining pool size
    trainRel, valRel = holdOut(n_rem, Pval_rel)

    # Map relative positions back to the original index labels
    valIdx   = trainValIdx[valRel]
    trainIdx = trainValIdx[trainRel]

    return (trainIdx, valIdx, testIdx)
end

#function trainClassANN

using Flux
using Flux.Losses

function _trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} =
        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} =
        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20,
    printLoss::Bool=false)

    # --- unpack & sanity checks ---
    (train_inputs, train_targets) = trainingDataset
    @assert size(train_inputs,1) == size(train_targets,1) "Train X and y must have same rows (samples)."

    (val_inputs, val_targets) = validationDataset
    @assert size(val_inputs,1) == size(val_targets,1) "Val X and y must have same rows (samples)."

    (test_inputs, test_targets) = testDataset
    @assert size(test_inputs,1) == size(test_targets,1) "Test X and y must have same rows (samples)."

    # --- model build ---

    ann = buildClassANN(size(train_inputs,2), topology, size(train_targets,2))

    # --- loss closure (Flux style) ---
    _loss = (model, x, y) -> (size(y,1) == 1 ?
        Flux.Losses.binarycrossentropy(model(x), y) :
        Flux.Losses.crossentropy(model(x), y))

    # --- tracking ---
    trainingLosses   = Float64[]
    validationLosses = Float64[]
    testLosses       = Float64[]

    # --- initial losses 
    trainL0 = _loss(ann, train_inputs', train_targets')
    push!(trainingLosses, trainL0)

    if !isempty(test_inputs)
        testL0 = _loss(ann, test_inputs', test_targets')
        push!(testLosses, testL0)  
    end
    if !isempty(val_inputs)
        valL0 = _loss(ann, val_inputs', val_targets')
        push!(validationLosses, valL0)
        if printLoss
            println("Loss epoch 0 -> training: ", round(trainL0; digits=4),
                    " | validation: ", round(valL0; digits=4),
                    (!isempty(test_inputs) ? " | test: " * string(round(testLosses[end]; digits=4)) : ""))
        end
    else
        if printLoss
            println("Loss epoch 0 -> training: ", round(trainL0; digits=4),
                    (!isempty(test_inputs) ? " | test: " * string(round(testLosses[end]; digits=4)) : ""))
        end
    end

    # --- optimizer ---
    opt_state = Flux.setup(Adam(learningRate), ann)

    # --- early stopping state ---

    monitor_validation = !isempty(val_inputs)
    best_metric = monitor_validation ? (isempty(validationLosses) ? Inf : validationLosses[end]) : trainingLosses[end]
    best_ann = deepcopy(ann)
    epochs_wo_improving = 0

    # --- training loop ---
    numEpoch = 0
    while (numEpoch < maxEpochs) && (trainingLosses[end] > minLoss) && (epochs_wo_improving < maxEpochsVal)
        numEpoch += 1

        
        data = [(train_inputs', train_targets')]
        Flux.train!(_loss, ann, data, opt_state)

        # compute losses post-update
        trainL = _loss(ann, train_inputs', train_targets')
        push!(trainingLosses, trainL)

        if !isempty(test_inputs)
            push!(testLosses, _loss(ann, test_inputs', test_targets'))
        end

        if !isempty(val_inputs)
            push!(validationLosses, _loss(ann, val_inputs', val_targets'))
            if printLoss
                println("Loss epoch ", numEpoch, " -> training: ", round(trainingLosses[end]; digits=4),
                        " | validation: ", round(validationLosses[end]; digits=4),
                        (!isempty(test_inputs) ? " | test: " * string(round(testLosses[end]; digits=4)) : ""))
            end
        else
            if printLoss
                println("Loss epoch ", numEpoch, " -> training: ", round(trainingLosses[end]; digits=4),
                        (!isempty(test_inputs) ? " | test: " * string(round(testLosses[end]; digits=4)) : ""))
            end
        end

        # --- early stopping
        current_metric = monitor_validation ? validationLosses[end] : trainingLosses[end]
        if current_metric < best_metric - eps() 
            best_metric = current_metric
            best_ann = deepcopy(ann)
            epochs_wo_improving = 0
        else
            epochs_wo_improving += 1
        end
    end

    return (best_ann, trainingLosses, validationLosses, testLosses)
end


function trainClassANN(topology::AbstractArray{<:Int,1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} =
                            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}} =
                            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),      
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        printLoss::Bool=false)

    (inputs, targets)     = trainingDataset
    (val_inputs, val_targets)   = validationDataset
    (test_inputs, test_targets) = testDataset

_trainClassANN(topology,
        (inputs, Float32.(reshape(targets, :, 1))); 
        validationDataset = (val_inputs, Float32.(reshape(val_targets, :, 1))),
        testDataset = (test_inputs, Float32.(reshape(test_targets, :, 1))),
        transferFunctions = transferFunctions,
        maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate,
        printLoss = printLoss)
end


#function confusionMatrix

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "outputs and targets must have the same length"

    # Confusion matrix counts
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    N = TP + TN + FP + FN
    acc = N == 0 ? 0.0 : (TP + TN) / N
    err = N == 0 ? 0.0 : (FP + FN) / N

    # Sensitivity (Recall) = TP / (TP + FN)
    # Specificity          = TN / (TN + FP)
    # Precision (PPV)      = TP / (TP + FP)
    # NPV                  = TN / (TN + FN)

    # Casi particolari richiesti dal testo:
    # - Se TUTTI i pattern sono TN: sensitivity e precision non sono definibili,
    #   ma vanno posti a 1 (sistema “corretto” sui negativi).
    all_TN = (TN == N) && (N > 0)
    # - Se TUTTI i pattern sono TP: specificity e NPV non sono definibili,
    #   ma vanno posti a 1 (sistema “corretto” sui positivi).
    all_TP = (TP == N) && (N > 0)

    # Sensitivity (recall)
    sens = if all_TN
        1.0
    else
        den = TP + FN
        den == 0 ? 0.0 : TP / den
    end

    # Specificity
    spec = if all_TP
        1.0
    else
        den = TN + FP
        den == 0 ? 0.0 : TN / den
    end

    # Precision (PPV)
    prec = if all_TN
        1.0
    else
        den = TP + FP
        den == 0 ? 0.0 : TP / den
    end

    # Negative Predictive Value (NPV)
    npv = if all_TP
        1.0
    else
        den = TN + FN
        den == 0 ? 0.0 : TN / den
    end

    # F-score: if sensitivity and precision are both 0, F = 0 
    F = (sens == 0.0 && prec == 0.0) ? 0.0 : (2 * sens * prec) / (sens + prec)

    CM = Array{Int64,2}(undef, 2, 2)
    # Row 1: Real Negative -> [TN FP]
    # Row 2: Real Positive -> [FN TP]
    CM[1,1] = TN; CM[1,2] = FP
    CM[2,1] = FN; CM[2,2] = TP

    return acc, err, sens, spec, prec, npv, F, CM
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    @assert length(outputs) == length(targets) "outputs and targets must have the same length"
    preds = outputs .>= threshold
    return confusionMatrix(preds, targets)
end

using Printf
function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    acc, err, sens, spec, prec, npv, F, CM = confusionMatrix(outputs, targets)

    println("Confusion Matrix")
    println("                Predicted")
    println("              |  Neg   Pos")
    println("Real | Neg    |  $(CM[1,1])     $(CM[1,2])")
    println("     | Pos    |  $(CM[2,1])     $(CM[2,2])")
    println()
    @printf("Accuracy:   %.4f\n", acc)
    @printf("Error rate: %.4f\n", err)
    @printf("Recall:     %.4f\n", sens)
    @printf("Specificity:%.4f\n", spec)
    @printf("Precision:  %.4f\n", prec)
    @printf("NPV:        %.4f\n", npv)
    @printf("F-score:    %.4f\n", F)

    return nothing
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Converti gli output reali in booleani usando la soglia
    outputs_bool = outputs .>= threshold
    # Chiama la prima funzione con gli output convertiti
    printConfusionMatrix(outputs_bool, targets)
    return nothing
end

function confusionMatrix(outputs::AbstractArray{Bool,2},
                         targets::AbstractArray{Bool,2};
                         weighted::Bool = true)

    @assert size(outputs) == size(targets) "outputs/targets must have same size"
    @assert size(targets, 2) > 2 "this function is for multiclass (>2 columns)"
    @assert all(sum(targets, dims=2) .== 1) "targets must be one-hot rows"
    @assert all(sum(outputs, dims=2) .== 1) "outputs must be one-hot rows"

    safediv(a, b) = b == 0 ? 0.0 : a / b

    N, C = size(targets)
    gt_counts   = vec(sum(targets, dims=1))
    pred_counts = vec(sum(outputs, dims=1))

    TP = [sum(outputs[:, i] .&  targets[:, i]) for i in 1:C]
    FP = [sum(outputs[:, i] .& .!targets[:, i]) for i in 1:C]
    FN = [sum(.!outputs[:, i] .& targets[:, i]) for i in 1:C]
    TN = [sum((.!outputs[:, i]) .& (.!targets[:, i])) for i in 1:C]

    sensitivity = [safediv(tp, tp + fn) for (tp, fn) in zip(TP, FN)]
    specificity = [safediv(tn, tn + fp) for (tn, fp) in zip(TN, FP)]
    ppv         = [safediv(tp, tp + fp) for (tp, fp) in zip(TP, FP)]
    npv         = [safediv(tn, tn + fn) for (tn, fn) in zip(TN, FN)]
    f1 = [safediv(2 * tp, 2 * tp + fp + fn) for (tp, fp, fn) in zip(TP, FP, FN)]

    CM = [sum(targets[:, i] .& outputs[:, j]) for i in 1:C, j in 1:C]  # rows=true, cols=pred

    if weighted
        wsum = sum(gt_counts)
        w = wsum == 0 ? fill(0.0, C) : gt_counts ./ wsum
        agg_sensitivity = sum(w .* sensitivity)
        agg_specificity = sum(w .* specificity)
        agg_ppv         = sum(w .* ppv)
        agg_npv         = sum(w .* npv)
        agg_f1          = sum(w .* f1)
    else
        # macro over classes present in either GT or predictions
        idx = findall((gt_counts .> 0) .| (pred_counts .> 0))
        agg_sensitivity = isempty(idx) ? 0.0 : mean(view(sensitivity, idx))
        agg_specificity = isempty(idx) ? 0.0 : mean(view(specificity, idx))
        agg_ppv         = isempty(idx) ? 0.0 : mean(view(ppv, idx))
        agg_npv         = isempty(idx) ? 0.0 : mean(view(npv, idx))
        agg_f1          = isempty(idx) ? 0.0 : mean(view(f1, idx))
    end

    correct   = sum(CM[i, i] for i in 1:C)
    total     = sum(CM)
    accuracy  = safediv(correct, total)
    errorrate = 1.0 - accuracy

    return (
        CM = CM,  # confusion matrix (true x predicted)
        per_class = (
            TP = TP, FP = FP, FN = FN, TN = TN,
            sensitivity = sensitivity, specificity = specificity,
            ppv = ppv, npv = npv, f1 = f1,
            gt_counts = gt_counts, pred_counts = pred_counts
        ),
        aggregated = (
            sensitivity = agg_sensitivity, specificity = agg_specificity,
            ppv = agg_ppv, npv = agg_npv, f1 = agg_f1, weighted = weighted
        ),
        accuracy = accuracy,
        errorrate = errorrate
    )
end


function confusionMatrix(outputs::AbstractArray{<:Real,2},
                         targets::AbstractArray{Bool,2};
                         threshold::Real = 0.5,
                         weighted::Bool = true)

    @assert size(outputs) == size(targets) "outputs and targets must have the same size"
    @assert size(targets,2) > 2 "Use the binary function for 1–2 columns; this is multiclass."

    outputs_bool = classifyOutputs(outputs; threshold=threshold)

    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1},
                         targets::AbstractArray{<:Any,1},
                         classes::AbstractArray{<:Any,1};
                         weighted::Bool = true)

    @assert length(outputs) == length(targets) "outputs and targets must have the same length"

    @assert all(lbl -> lbl in classes, unique(vcat(outputs, targets))) 
        "All labels in outputs/targets must be included in `classes`"

    ŷ = oneHotEncoding(outputs, classes)   # Bool matrix: N × C
    Y = oneHotEncoding(targets, classes)   # Bool matrix: N × C

    return confusionMatrix(ŷ, Y; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1},
                         targets::AbstractArray{<:Any,1};
                         weighted::Bool = true)

    @assert length(outputs) == length(targets) "outputs and targets must have the same length"

    classes = unique(vcat(targets, outputs))

    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end


function printConfusionMatrix(outputs::AbstractArray{Bool,2},
                              targets::AbstractArray{Bool,2};
                              weighted::Bool = true)

    res = confusionMatrix(outputs, targets; weighted=weighted)
    C = size(targets, 2)
    classes = collect(1:C)                 # fallback labels 1..C

    # -- header line
    print("pred\\true")
    for j in 1:C
        print('\t', classes[j])
    end
    println()

    # -- each row of CM (predicted i, true j)
    for i in 1:C
        print(classes[i])
        for j in 1:C
            print('\t', res.CM[i, j])
        end
        println()
    end

    # -- summary lines
    println("accuracy = ", res.accuracy, "   error rate = ", res.errorrate)
    agg = res.aggregated
    println("aggregated (", weighted ? "weighted" : "macro", "): ",
            "sens=", agg.sensitivity, "  spec=", agg.specificity,
            "  ppv=", agg.ppv, "  npv=", agg.npv, "  f1=", agg.f1)
end


function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
                              targets::AbstractArray{Bool,2};
                              weighted::Bool = true)

    res = confusionMatrix(outputs, targets; weighted=weighted)
    C = size(targets, 2)
    classes = collect(1:C)

    print("pred\\true")
    for j in 1:C
        print('\t', classes[j])
    end
    println()

    for i in 1:C
        print(classes[i])
        for j in 1:C
            print('\t', res.CM[i, j])
        end
        println()
    end

    println("accuracy = ", res.accuracy, "   error rate = ", res.errorrate)
    agg = res.aggregated
    println("aggregated (", weighted ? "weighted" : "macro", "): ",
            "sens=", agg.sensitivity, "  spec=", agg.specificity,
            "  ppv=", agg.ppv, "  npv=", agg.npv, "  f1=", agg.f1)
end


function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
                              targets::AbstractArray{<:Any,1},
                              classes::AbstractArray{<:Any,1};
                              weighted::Bool = true)

    res = confusionMatrix(outputs, targets, classes; weighted=weighted)
    C = length(classes)

    print("pred\\true")
    for j in 1:C
        print('\t', string(classes[j]))
    end
    println()

    for i in 1:C
        print(string(classes[i]))
        for j in 1:C
            print('\t', res.CM[i, j])
        end
        println()
    end

    println("accuracy = ", res.accuracy, "   error rate = ", res.errorrate)
    agg = res.aggregated
    println("aggregated (", weighted ? "weighted" : "macro", "): ",
            "sens=", agg.sensitivity, "  spec=", agg.specificity,
            "  ppv=", agg.ppv, "  npv=", agg.npv, "  f1=", agg.f1)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
                              targets::AbstractArray{<:Any,1};
                              weighted::Bool = true)

    classes = unique(vcat(targets, outputs))
    res = confusionMatrix(outputs, targets, classes; weighted=weighted)
    C = length(classes)

    print("pred\\true")
    for j in 1:C
        print('\t', string(classes[j]))
    end
    println()

    for i in 1:C
        print(string(classes[i]))
        for j in 1:C
            print('\t', res.CM[i, j])
        end
        println()
    end

    println("accuracy = ", res.accuracy, "   error rate = ", res.errorrate)
    agg = res.aggregated
    println("aggregated (", weighted ? "weighted" : "macro", "): ",
            "sens=", agg.sensitivity, "  spec=", agg.specificity,
            "  ppv=", agg.ppv, "  npv=", agg.npv, "  f1=", agg.f1)
end

#functionANNCrossValidation

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1};
        numExecutions::Int=50,
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        validationRatio::Real=0, maxEpochsVal::Int=20)
    
    inputs, targets = dataset

    # 1. Extract class labels
    classes = unique(targets)
    numClasses = length(classes)

    # 2. One-hot encode targets
    targets_ohe = oneHotEncoding(targets, classes)

    # 3. Determine numFolds
    numFolds = maximum(crossValidationIndices)

    # 4. Create vectors to store results for each fold
    fold_accuracies = zeros(Float64, numFolds)
    fold_error_rates = zeros(Float64, numFolds)
    fold_sensitivities = zeros(Float64, numFolds)
    fold_specificities = zeros(Float64, numFolds)
    fold_ppvs = zeros(Float64, numFolds)
    fold_npvs = zeros(Float64, numFolds)
    fold_f1s = zeros(Float64, numFolds)

    # 5. Initialize global confusion matrix accumulator
    global_confusion_matrix = zeros(Int64, numClasses, numClasses)

    # --- Outer loop: Iterate over each fold ---
    for k in 1:numFolds
        # Extract train and test subsets for this fold
        test_indices = (crossValidationIndices .== k)
        train_indices = .!test_indices
        
        train_inputs = inputs[train_indices, :]
        train_targets = targets_ohe[train_indices, :]
        
        test_inputs = inputs[test_indices, :]
        test_targets = targets_ohe[test_indices, :]

        # --- Inner loop: Iterate over numExecutions ---
        
        # 1. Initialize vectors for execution metrics
        exec_accuracies = zeros(Float64, numExecutions)
        exec_error_rates = zeros(Float64, numExecutions)
        exec_sensitivities = zeros(Float64, numExecutions)
        exec_specificities = zeros(Float64, numExecutions)
        exec_ppvs = zeros(Float64, numExecutions)
        exec_npvs = zeros(Float64, numExecutions)
        exec_f1s = zeros(Float64, numExecutions)

        # 2. Create 3D array for confusion matrices
        exec_confusion_matrices = zeros(Int64, numClasses, numClasses, numExecutions)

        # 3. For each execution
        for exec in 1:numExecutions
            # Define validation set (if needed)
            local val_inputs, val_targets
            local current_train_inputs, current_train_targets
            
            if validationRatio > 0
                # Split the fold's training set into new train/validation sets
                N_train_fold = size(train_inputs, 1)
                (current_train_idx, val_idx) = holdOut(N_train_fold, validationRatio)
                
                current_train_inputs = train_inputs[current_train_idx, :]
                current_train_targets = train_targets[current_train_idx, :]
                val_inputs = train_inputs[val_idx, :]
                val_targets = train_targets[val_idx, :]
            else
                # No validation, use the whole fold's training set
                current_train_inputs = train_inputs
                current_train_targets = train_targets
                # Create empty validation sets as expected by trainClassANN 
                val_inputs = Array{eltype(inputs), 2}(undef, 0, size(inputs, 2))
                val_targets = falses(0, numClasses)
            end
            
            # Train the network
            (ann, _, _, _) = _trainClassANN(topology,
                (current_train_inputs, current_train_targets);
                validationDataset = (val_inputs, val_targets),
                testDataset = (test_inputs, test_targets),
                transferFunctions = transferFunctions,
                maxEpochs = maxEpochs,
                minLoss = minLoss,
                learningRate = learningRate,
                maxEpochsVal = maxEpochsVal,
                printLoss = false)
            
            # Evaluate on the test set
            test_outputs_real = ann(test_inputs')'
              
            if (numClasses == 2)
                
                (acc, err, sens, spec, prec, npv, F_score, cm) = confusionMatrix(test_outputs_real[:, 1], test_targets[:, 1]; threshold=0.5)
                
                exec_accuracies[exec] = acc
                exec_error_rates[exec] = err
                exec_sensitivities[exec] = sens
                exec_specificities[exec] = spec
                exec_ppvs[exec] = prec
                exec_npvs[exec] = npv
                exec_f1s[exec] = F_score
                exec_confusion_matrices[:, :, exec] = cm

            else
                
                cm_results = confusionMatrix(test_outputs_real, test_targets; weighted=true)
                
                exec_accuracies[exec] = cm_results.accuracy
                exec_error_rates[exec] = cm_results.errorrate
                exec_sensitivities[exec] = cm_results.aggregated.sensitivity
                exec_specificities[exec] = cm_results.aggregated.specificity
                exec_ppvs[exec] = cm_results.aggregated.ppv
                exec_npvs[exec] = cm_results.aggregated.npv
                exec_f1s[exec] = cm_results.aggregated.f1
                exec_confusion_matrices[:, :, exec] = cm_results.CM
            end

        end
        
        # 4. After all executions for this fold
        
        # Compute average metrics for the fold
        fold_accuracies[k] = mean(exec_accuracies)
        fold_error_rates[k] = mean(exec_error_rates)
        fold_sensitivities[k] = mean(exec_sensitivities)
        fold_specificities[k] = mean(exec_specificities)
        fold_ppvs[k] = mean(exec_ppvs)
        fold_npvs[k] = mean(exec_npvs)
        fold_f1s[k] = mean(exec_f1s)
        
        # Compute SUM confusion matrix for the fold
        sum_cm_fold = sum(exec_confusion_matrices, dims=3)
        
        # Add to the global confusion matrix accumulator
        global_confusion_matrix .+= sum_cm_fold[:, :, 1]
    end

    # --- After all folds ---
    
    # Compute mean and standard deviation for each metric across folds
    acc_stats = (mean(fold_accuracies), std(fold_accuracies))
    err_stats = (mean(fold_error_rates), std(fold_error_rates))
    sens_stats = (mean(fold_sensitivities), std(fold_sensitivities))
    spec_stats = (mean(fold_specificities), std(fold_specificities))
    ppv_stats = (mean(fold_ppvs), std(fold_ppvs))
    npv_stats = (mean(fold_npvs), std(fold_npvs))
    f1_stats = (mean(fold_f1s), std(fold_f1s))

    # Return the tuple with all results
    return (acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, global_confusion_matrix)
end


function ANNCrossValidation(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1};
        numExecutions::Int=50,
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        validationRatio::Real=0, maxEpochsVal::Int=20)
    
    inputs, targets = dataset

    # 1. Extract class labels
    classes = unique(targets)
    numClasses = length(classes)

    # 2. One-hot encode targets
    targets_ohe = oneHotEncoding(targets, classes)

    # 3. Determine numFolds
    numFolds = maximum(crossValidationIndices)

    # 4. Create vectors to store results for each fold
    fold_accuracies = zeros(Float64, numFolds)
    fold_error_rates = zeros(Float64, numFolds)
    fold_sensitivities = zeros(Float64, numFolds)
    fold_specificities = zeros(Float64, numFolds)
    fold_ppvs = zeros(Float64, numFolds)
    fold_npvs = zeros(Float64, numFolds)
    fold_f1s = zeros(Float64, numFolds)

    # 5. Initialize global confusion matrix accumulator
    global_confusion_matrix = zeros(Int64, numClasses, numClasses)

    # --- Outer loop: Iterate over each fold ---
    for k in 1:numFolds
        # Extract train and test subsets for this fold
        test_indices = (crossValidationIndices .== k)
        train_indices = .!test_indices
        
        train_inputs = inputs[train_indices, :]
        train_targets = targets_ohe[train_indices, :]
        
        test_inputs = inputs[test_indices, :]
        test_targets = targets_ohe[test_indices, :]

        # --- Inner loop: Iterate over numExecutions ---
        
        # 1. Initialize vectors for execution metrics
        exec_accuracies = zeros(Float64, numExecutions)
        exec_error_rates = zeros(Float64, numExecutions)
        exec_sensitivities = zeros(Float64, numExecutions)
        exec_specificities = zeros(Float64, numExecutions)
        exec_ppvs = zeros(Float64, numExecutions)
        exec_npvs = zeros(Float64, numExecutions)
        exec_f1s = zeros(Float64, numExecutions)

        # 2. Create 3D array for confusion matrices
        exec_confusion_matrices = zeros(Int64, numClasses, numClasses, numExecutions)

        # 3. For each execution
        for exec in 1:numExecutions
            # Define validation set (if needed)
            local val_inputs, val_targets
            local current_train_inputs, current_train_targets
            
            if validationRatio > 0
                # Split the fold's training set into new train/validation sets
                N_train_fold = size(train_inputs, 1)
                (current_train_idx, val_idx) = holdOut(N_train_fold, validationRatio)
                
                current_train_inputs = train_inputs[current_train_idx, :]
                current_train_targets = train_targets[current_train_idx, :]
                val_inputs = train_inputs[val_idx, :]
                val_targets = train_targets[val_idx, :]
            else
                # No validation, use the whole fold's training set
                current_train_inputs = train_inputs
                current_train_targets = train_targets
                # Create empty validation sets as expected by trainClassANN 
                val_inputs = Array{eltype(inputs), 2}(undef, 0, size(inputs, 2))
                val_targets = falses(0, numClasses)
            end
            
            # Train the network
            (ann, _, _, _) = _trainClassANN(topology,
                (current_train_inputs, current_train_targets);
                validationDataset = (val_inputs, val_targets),
                testDataset = (test_inputs, test_targets),
                transferFunctions = transferFunctions,
                maxEpochs = maxEpochs,
                minLoss = minLoss,
                learningRate = learningRate,
                maxEpochsVal = maxEpochsVal,
                printLoss = false)
            
            # Evaluate on the test set
            test_outputs_real = ann(test_inputs')'
              
            if (numClasses == 2)
                
                (acc, err, sens, spec, prec, npv, F_score, cm) = confusionMatrix(test_outputs_real[:, 1], test_targets[:, 1]; threshold=0.5)
                
                exec_accuracies[exec] = acc
                exec_error_rates[exec] = err
                exec_sensitivities[exec] = sens
                exec_specificities[exec] = spec
                exec_ppvs[exec] = prec
                exec_npvs[exec] = npv
                exec_f1s[exec] = F_score
                exec_confusion_matrices[:, :, exec] = cm

            else
                
                cm_results = confusionMatrix(test_outputs_real, test_targets; weighted=true)
                
                exec_accuracies[exec] = cm_results.accuracy
                exec_error_rates[exec] = cm_results.errorrate
                exec_sensitivities[exec] = cm_results.aggregated.sensitivity
                exec_specificities[exec] = cm_results.aggregated.specificity
                exec_ppvs[exec] = cm_results.aggregated.ppv
                exec_npvs[exec] = cm_results.aggregated.npv
                exec_f1s[exec] = cm_results.aggregated.f1
                exec_confusion_matrices[:, :, exec] = cm_results.CM
            end

        end
        
        # 4. After all executions for this fold
        
        # Compute average metrics for the fold
        fold_accuracies[k] = mean(exec_accuracies)
        fold_error_rates[k] = mean(exec_error_rates)
        fold_sensitivities[k] = mean(exec_sensitivities)
        fold_specificities[k] = mean(exec_specificities)
        fold_ppvs[k] = mean(exec_ppvs)
        fold_npvs[k] = mean(exec_npvs)
        fold_f1s[k] = mean(exec_f1s)
        
        # Compute SUM confusion matrix for the fold
        sum_cm_fold = sum(exec_confusion_matrices, dims=3)
        
        # Add to the global confusion matrix accumulator
        global_confusion_matrix .+= sum_cm_fold[:, :, 1]
    end

    # --- After all folds ---
    
    # Compute mean and standard deviation for each metric across folds
    acc_stats = (mean(fold_accuracies), std(fold_accuracies))
    err_stats = (mean(fold_error_rates), std(fold_error_rates))
    sens_stats = (mean(fold_sensitivities), std(fold_sensitivities))
    spec_stats = (mean(fold_specificities), std(fold_specificities))
    ppv_stats = (mean(fold_ppvs), std(fold_ppvs))
    npv_stats = (mean(fold_npvs), std(fold_npvs))
    f1_stats = (mean(fold_f1s), std(fold_f1s))

    # Return the tuple with all results
    return (acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, global_confusion_matrix)
end


using Statistics # for mean, std
using Random # for MersenneTwister

"""
    modelCrossValidation(modelType, modelHyperparameters, dataset, crossValidationIndices)

Performs k-fold cross-validation for a specified model type (ANN, SVC, 
DecisionTree, or kNN) using the provided data and fold indices.

Returns an 8-tuple containing:
(acc, err, sens, spec, ppv, npv, f1, global_cm)
where each metric is a (mean, std) tuple and global_cm is the summed confusion matrix.
"""
function modelCrossValidation(
        modelType::Symbol, modelHyperparameters::Dict,
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1})

    # Unpack dataset
    inputs, targets = dataset

    # --- Handle ANN case separately ---
    if modelType == :ANN
        # Call the ANNCrossValidation function from utils.jl [cite: 102]
        
        # Extract/default ANN hyperparameters
        topology = get(modelHyperparameters, "topology", [5, 3]) # Default topology
        learningRate = get(modelHyperparameters, "learningRate", 0.01)
        validationRatio = get(modelHyperparameters, "validationRatio", 0.0)
        numExecutions = get(modelHyperparameters, "numExecutions", 50)
        maxEpochs = get(modelHyperparameters, "maxEpochs", 1000)
        maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20)
        
        # This function already returns the required 8-tuple [cite: 102, 118]
        return ANNCrossValidation(
            topology,
            dataset, # Pass the original dataset (inputs, targets)
            crossValidationIndices;
            numExecutions = numExecutions,
            maxEpochs = maxEpochs,
            learningRate = learningRate,
            validationRatio = validationRatio,
            maxEpochsVal = maxEpochsVal
        )
    end

    # --- MLJ Models Logic ---

    # Convert targets to string (as instructed in the notebook)
    targets_str = string.(targets)
    classes = unique(targets_str)
    numClasses = length(classes)
    numFolds = maximum(crossValidationIndices)

    # Create vectors to store metrics for each fold
    accuracies = zeros(Float64, numFolds)
    error_rates = zeros(Float64, numFolds)
    sensitivities = zeros(Float64, numFolds)
    specificities = zeros(Float64, numFolds)
    ppvs = zeros(Float64, numFolds)
    npvs = zeros(Float64, numFolds)
    f1s = zeros(Float64, numFolds)

    # Create accumulator for confusion matrix
    globalConfusionMatrix = zeros(Int64, numClasses, numClasses)

    # --- Cross-validation loop ---
    for k in 1:numFolds
        # 1. Extract training and test data for this fold
        testIndices = (crossValidationIndices .== k)
        trainIndices = .!testIndices

        Xtrain = inputs[trainIndices, :]
        ytrain = targets_str[trainIndices] # Use string targets
        
        Xtest = inputs[testIndices, :]
        ytest = targets_str[testIndices] # Use string targets

        # 2. Create the model instance
        local model # Define model in local scope
        
        if modelType == :SVC
            # Parse kernel string
            kernel_str = get(modelHyperparameters, "kernel", "rbf")
            local kernel_func
            if kernel_str == "linear"
                kernel_func = LIBSVM.Kernel.Linear
            elseif kernel_str == "poly"
                kernel_func = LIBSVM.Kernel.Polynomial
            elseif kernel_str == "sigmoid"
                kernel_func = LIBSVM.Kernel.Sigmoid
            else # default to rbf
                kernel_func = LIBSVM.Kernel.RadialBasis
            end

            # Get all params, with defaults, and cast to required types
            C = Float64(get(modelHyperparameters, "C", 1.0))
            gamma = Float64(get(modelHyperparameters, "gamma", 1.0))
            degree = Int32(get(modelHyperparameters, "degree", 3))
            coef0 = Float64(get(modelHyperparameters, "coef0", 0.0))

            model = SVMClassifier(
                kernel=kernel_func,
                cost=C,
                gamma=gamma,
                degree=degree,
                coef0=coef0
            )

        elseif modelType == :DecisionTreeClassifier
            max_depth = get(modelHyperparameters, "max_depth", -1) # -1 for unlimited
            # Per instructions, force RNG for reproducibility
            rng = Random.MersenneTwister(1) 
            model = DTClassifier(max_depth=max_depth, rng=rng)

        elseif modelType == :KNeighborsClassifier
            k_neighbors = get(modelHyperparameters, "n_neighbors", 3)
            model = kNNClassifier(K=k_neighbors)
        
        else
            error("Unknown modelType: $modelType")
        end

        # 3. Create and fit the machine
        # Note: ytrain must be wrapped in `categorical` for MLJ classification
        mach = machine(model, MLJ.table(Xtrain), categorical(ytrain))
        MLJ.fit!(mach, verbosity=0)

        # 4. Perform predictions on the test set
        predictions_raw = MLJ.predict(mach, MLJ.table(Xtest))
        
        local ŷ # To store the final predicted labels
        if modelType == :SVC
            # SVM returns a CategoricalArray of labels directly
            ŷ = predictions_raw 
        else
            # DT and kNN return UnivariateFiniteArray (probabilities)
            # We must get the most likely class (the mode)
            ŷ = mode.(predictions_raw)
        end
        
        # 5. Calculate and store metrics
        # Use the confusionMatrix function from utils.jl that takes class labels
        cm_results = confusionMatrix(ŷ, ytest, classes; weighted=true) # 
        
        # Store metrics
        accuracies[k] = cm_results.accuracy
        error_rates[k] = cm_results.errorrate
        # Use aggregated (weighted) metrics
        sensitivities[k] = cm_results.aggregated.sensitivity
        specificities[k] = cm_results.aggregated.specificity
        ppvs[k] = cm_results.aggregated.ppv
        npvs[k] = cm_results.aggregated.npv
        f1s[k] = cm_results.aggregated.f1
        
        # Add to global confusion matrix
        globalConfusionMatrix .+= cm_results.CM
    end # end fold loop

    # 6. Calculate final statistics (mean and std) across folds
    acc_stats = (mean(accuracies), std(accuracies))
    err_stats = (mean(error_rates), std(error_rates))
    sens_stats = (mean(sensitivities), std(sensitivities))
    spec_stats = (mean(specificities), std(specificities))
    ppv_stats = (mean(ppvs), std(ppvs))
    npv_stats = (mean(npvs), std(npvs))
    f1_stats = (mean(f1s), std(f1s))
    
    # 7. Return the 8-tuple as specified
    return (acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, globalConfusionMatrix)
end

# ============================================================================
#     MODEL CROSS-VALIDATION (Extended for Project - Multiclass Support)
# ============================================================================

# Note: This function extends the course utilities to support
# SVMs, Decision Trees, and kNN using MLJ, following the same
# pattern as the original modelCrossValidation function.

# Load MLJ models
using MLJ
using LIBSVM
SVMClassifier = @load SVC pkg=LIBSVM
kNNClassifier = @load KNNClassifier pkg=NearestNeighborModels  
DTClassifier = @load DecisionTreeClassifier pkg=DecisionTree

function modelCrossValidation(
        modelType::Symbol, 
        modelHyperparameters::Dict,
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Int,1}},
        crossValidationIndices::Array{Int64,1})

    inputs, targets = dataset

    # --- Handle ANN case (as in original course code) ---
    if modelType == :ANN
        topology = get(modelHyperparameters, "topology", [5, 3])
        learningRate = get(modelHyperparameters, "learningRate", 0.003)
        validationRatio = get(modelHyperparameters, "validationRatio", 0.1)
        numExecutions = get(modelHyperparameters, "numExecutions", 1)
        maxEpochs = get(modelHyperparameters, "maxEpochs", 800)
        maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 25)
        
        return ANNCrossValidation(
            topology,
            dataset,
            crossValidationIndices;
            numExecutions = numExecutions,
            maxEpochs = maxEpochs,
            learningRate = learningRate,
            validationRatio = validationRatio,
            maxEpochsVal = maxEpochsVal
        )
    end

    # --- MLJ Models Logic (SVM, DT, kNN) ---
    
    targets_str = string.(targets)
    classes = sort(unique(targets_str))
    numClasses = length(classes)
    numFolds = maximum(crossValidationIndices)

    # Metrics storage
    accuracies = zeros(Float64, numFolds)
    error_rates = zeros(Float64, numFolds)
    sensitivities = zeros(Float64, numFolds)
    specificities = zeros(Float64, numFolds)
    ppvs = zeros(Float64, numFolds)
    npvs = zeros(Float64, numFolds)
    f1s = zeros(Float64, numFolds)
    
    globalConfusionMatrix = zeros(Int64, numClasses, numClasses)

    # Cross-validation loop
    for k in 1:numFolds
        testIndices = (crossValidationIndices .== k)
        trainIndices = .!testIndices

        Xtrain = inputs[trainIndices, :]
        ytrain = targets_str[trainIndices]
        
        Xtest = inputs[testIndices, :]
        ytest = targets_str[testIndices]

        # Normalize
        normParams = calculateMinMaxNormalizationParameters(Xtrain)
        Xtrain_norm = normalizeMinMax(Xtrain, normParams)
        Xtest_norm = normalizeMinMax(Xtest, normParams)

        # Create model
        local model
        
        if modelType == :SVC
            kernel_str = get(modelHyperparameters, "kernel", "rbf")
            local kernel_func
            if kernel_str == "linear"
                kernel_func = LIBSVM.Kernel.Linear
            elseif kernel_str == "poly"
                kernel_func = LIBSVM.Kernel.Polynomial
            elseif kernel_str == "sigmoid"
                kernel_func = LIBSVM.Kernel.Sigmoid
            else
                kernel_func = LIBSVM.Kernel.RadialBasis
            end

            C = Float64(get(modelHyperparameters, "C", 1.0))
            gamma = Float64(get(modelHyperparameters, "gamma", 0.125))
            degree = Int32(get(modelHyperparameters, "degree", 3))
            coef0 = Float64(get(modelHyperparameters, "coef0", 0.0))

            model = SVMClassifier(
                kernel=kernel_func,
                cost=C,
                gamma=gamma,
                degree=degree,
                coef0=coef0
            )

        elseif modelType == :DecisionTreeClassifier
            max_depth = get(modelHyperparameters, "max_depth", -1)
            rng = Random.MersenneTwister(42)
            model = DTClassifier(max_depth=max_depth, rng=rng)

        elseif modelType == :KNeighborsClassifier
            k_neighbors = get(modelHyperparameters, "n_neighbors", 5)
            model = kNNClassifier(K=k_neighbors)
        
        else
            error("Unknown modelType: $modelType")
        end

        # Fit model
        mach = machine(model, MLJ.table(Xtrain_norm), categorical(ytrain))
        MLJ.fit!(mach, verbosity=0)

        # Predict
        predictions_raw = MLJ.predict(mach, MLJ.table(Xtest_norm))
        
        local ŷ
        if modelType == :SVC
            ŷ = predictions_raw
        else
            ŷ = mode.(predictions_raw)
        end
        
        # Calculate metrics using confusionMatrix from utils
        cm_results = confusionMatrix(ŷ, ytest, classes; weighted=true)
        
        accuracies[k] = cm_results.accuracy
        error_rates[k] = cm_results.errorrate
        sensitivities[k] = cm_results.aggregated.sensitivity
        specificities[k] = cm_results.aggregated.specificity
        ppvs[k] = cm_results.aggregated.ppv
        npvs[k] = cm_results.aggregated.npv
        f1s[k] = cm_results.aggregated.f1
        
        globalConfusionMatrix .+= cm_results.CM
    end

    # Calculate statistics
    acc_stats = (mean(accuracies), std(accuracies))
    err_stats = (mean(error_rates), std(error_rates))
    sens_stats = (mean(sensitivities), std(sensitivities))
    spec_stats = (mean(specificities), std(specificities))
    ppv_stats = (mean(ppvs), std(ppvs))
    npv_stats = (mean(npvs), std(npvs))
    f1_stats = (mean(f1s), std(f1s))
    
    # Return 8-tuple (same format as course function)
    return (acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, globalConfusionMatrix)
end

# ============================================================================
#     CROSS-VALIDATION INDEX GENERATION (Stratified)
# ============================================================================

function crossvalidation(targets::AbstractArray{Int,1}, k::Int64)
    """
    Create stratified k-fold cross-validation indices
    Returns: array of fold assignments (1 to k) for each sample
    """
    indices = zeros(Int, length(targets))
    
    # For each class, assign fold indices
    for class_label in unique(targets)
        class_mask = targets .== class_label
        n_class = sum(class_mask)
        
        # Create k folds for this class
        class_indices = repeat(1:k, Int(ceil(n_class/k)))[1:n_class]
        shuffle!(class_indices)
        
        # Assign to main indices array
        indices[class_mask] .= class_indices
    end
    
    return indices
end
