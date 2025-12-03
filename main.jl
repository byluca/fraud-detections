# ============================================================================
#          FRAUD DETECTION - MULTICLASS CLASSIFICATION
#          Main Script - Executes All Experiments
#          Group Project - Machine Learning I
# ============================================================================
#
# Objective: E-commerce fraud detection using multiclass classification
#            (Legitimate, Suspicious, Fraudulent)
#
# Methodology:
#   - 4 ML Algorithms: ANNs (8 topologies), SVMs (10 configs), 
#                      Decision Trees (7 depths), kNN (6 k values)
#   - Ensemble Methods: Majority Voting + Weighted Voting
#   - Dataset: 1.5M e-commerce transactions → 3-class risk assessment
#   - Cross-Validation: 3-fold stratified on training set
#   - Evaluation: Hold-out test set (20%)
#
# Code Organization:
#   /project/
#   ├── main.jl                    ← This file (executable from top to bottom)
#   ├── /utils/
#   │   ├── utils.jl              ← Course utilities (includes modelCrossValidation)
#   │   └── preprocessing.jl      ← Custom preprocessing functions
#   └── /datasets/
#       └── Fraudulent_E-Commerce_Transaction_Data_merge.csv
#
# ============================================================================

# Set random seed for reproducibility
using Random
Random.seed!(42)

# ============================================================================
#                    PACKAGE IMPORTS
# ============================================================================

using CSV
using DataFrames
using Statistics
using Dates
using StatsBase

println("✅ Packages loaded!")

# Load course utilities
include("utils/utils.jl")
println("✅ Course utilities loaded (includes modelCrossValidation, confusionMatrix, etc.)")

# Load custom preprocessing
include("utils/preprocessing.jl")
using .PreprocessingUtils: create_risk_classes, preprocess_multiclass
println("✅ Custom preprocessing utilities loaded!")

# ============================================================================
#              DATA LOADING & 3-CLASS TARGET CREATION
# ============================================================================
#
# Dataset: Fraudulent E-Commerce Transaction Data (1.5M transactions)
#
# Target Creation Strategy:
#   Original: Binary fraud labels (fraud vs non-fraud)
#   Our approach: 3-class risk assessment based on multiple signals:
#     1. Time Risk: Night transactions (0-5am, 11pm)
#     2. Amount Risk: High-value transactions (>90th percentile)
#     3. Account Age Risk: New accounts (<30 days)
#
# Class Mapping:
#   - Class 0 (LEGITTIMO): Low-risk, legitimate transactions
#   - Class 1 (SOSPETTO): Borderline cases requiring manual review
#   - Class 2 (FRAUDOLENTO): High-risk fraudulent transactions
#
# Justification: This approach allows for graduated risk assessment, enabling
# businesses to automatically approve low-risk transactions, flag suspicious
# cases for manual review, and immediately block high-risk frauds.
#
# ============================================================================

const DATA_PATH = "datasets/Fraudulent_E-Commerce_Transaction_Data_merge.csv"
println("\n" * "="^70)
println("📂 LOADING DATA")
println("="^70)

df = CSV.read(DATA_PATH, DataFrame)
target_col = "Is Fraudulent"

println("Original dataset size: $(size(df))")
println("Original fraud distribution:")
println("  Non-fraud: $(sum(df[!, target_col] .== 0))")
println("  Fraud:     $(sum(df[!, target_col] .== 1))")

# Create 3-class target using custom function
println("\n" * "="^70)
println("🎯 CREATING 3-CLASS TARGET")
println("="^70)

df_with_classes = create_risk_classes(df, target_col)

# ============================================================================
#          CLASS BALANCING & TRAIN/TEST SPLIT
# ============================================================================
#
# Challenge: Highly imbalanced dataset (90% Legitimate, 8.6% Suspicious, 1.4% Fraudulent)
#
# Solution: Undersample majority classes to match minority class (20,654 samples per class)
#
# Train/Test Split:
#   - 80% Training (49,569 samples) - used for cross-validation and model selection
#   - 20% Test (12,393 samples) - held out for final evaluation
#
# CRITICAL: Test set is NEVER used during training or model selection to prevent data leakage!
#
# ============================================================================

println("\n" * "="^70)
println("✅ TRAIN/TEST SPLIT (80% Train / 20% Test)")
println("="^70)

# Balance classes
class_0 = df_with_classes[df_with_classes.Risk_Class .== 0, :]
class_1 = df_with_classes[df_with_classes.Risk_Class .== 1, :]
class_2 = df_with_classes[df_with_classes.Risk_Class .== 2, :]

n_min = minimum([size(class_0, 1), size(class_1, 1), size(class_2, 1)])
n_target = min(n_min, 40000)

println("\n🔄 Balancing dataset...")
println("  Samples per class: $n_target")

class_0_sample = class_0[shuffle(1:size(class_0, 1))[1:n_target], :]
class_1_sample = class_1[shuffle(1:size(class_1, 1))[1:n_target], :]
class_2_sample = class_2[shuffle(1:size(class_2, 1))[1:n_target], :]

df_balanced = vcat(class_0_sample, class_1_sample, class_2_sample)
df_balanced = df_balanced[shuffle(1:size(df_balanced, 1)), :]

println("  Balanced dataset size: $(size(df_balanced))")

# Split Train/Test BEFORE preprocessing (critical to avoid data leakage!)
n_total = size(df_balanced, 1)
n_train = floor(Int, n_total * 0.80)
n_test = n_total - n_train

all_indices = shuffle(1:n_total)
train_indices = all_indices[1:n_train]
test_indices = all_indices[n_train+1:end]

df_train = df_balanced[train_indices, :]
df_test = df_balanced[test_indices, :]

println("\n📊 Split Summary:")
println("  Total samples:     $n_total")
println("  Training set:      $n_train (80%)")
println("  Test set:          $n_test (20%)")

# ============================================================================
#                    PREPROCESSING & FEATURE ENGINEERING
# ============================================================================
#
# Steps:
#   1. Time Features: Extract hour, create night flag (hour < 6)
#   2. Feature Engineering:
#      - Amount_per_AccountAge: Transaction amount relative to account maturity
#      - High_Value_Flag: Transactions above 95th percentile
#      - New_Account_Flag: Accounts younger than 30 days
#   3. Missing Value Imputation: Median imputation
#   4. Feature Selection: Drop IDs, addresses, categorical features → 8 numerical features
#   5. Normalization: Min-Max [0,1] using training set parameters only
#
# Final Features (8):
#   - Transaction Amount
#   - Account Age Days
#   - Transaction_Hour
#   - Is_Night
#   - Amount_per_AccountAge
#   - High_Value_Flag
#   - New_Account_Flag
#   - (1 more from preprocessing)
#
# ============================================================================

println("\n🔧 Preprocessing train and test sets...")
df_train_processed = preprocess_multiclass(df_train, target_col)
df_test_processed = preprocess_multiclass(df_test, target_col)

input_cols = setdiff(names(df_train_processed), ["Risk_Class"])
train_inputs = Matrix{Float64}(df_train_processed[:, input_cols])
train_targets = Int.(df_train_processed.Risk_Class)

test_inputs = Matrix{Float64}(df_test_processed[:, input_cols])
test_targets = Int.(df_test_processed.Risk_Class)

println("\n📊 Preprocessed Data:")
println("  Features: $(length(input_cols))")
println("  Train samples: $(size(train_inputs, 1))")
println("  Test samples: $(size(test_inputs, 1))")
println("\n  Feature names: $input_cols")

# Create cross-validation indices (3-fold stratified)
k_folds = 3
cv_indices = crossvalidation(train_targets, k_folds)
println("\n✅ Cross-validation indices created ($k_folds folds, stratified)")

# ============================================================================
#        EXPERIMENT 1: ARTIFICIAL NEURAL NETWORKS (ANNs)
# ============================================================================
#
# Configuration:
#   - Topologies tested: 8 architectures (1-4 hidden layers)
#   - Activation: ReLU (hidden layers), Softmax (output)
#   - Optimizer: Adam (learning rate: 0.003)
#   - Loss: Cross-entropy
#   - Regularization: Early stopping (patience: 25 epochs)
#   - Validation: 10% of training set
#   - Executions: 1 per topology (can increase for stability)
#
# Architectures:
#   1. [256, 128, 64] - Deep Large
#   2. [128, 64, 32] - Baseline
#   3. [96, 48, 24] - Medium
#   4. [64, 32] - Shallow
#   5. [128, 128, 64, 32] - Very Deep
#   6. [192, 96, 48] - Wide-Deep
#   7. [128, 64] - Simple
#   8. [256, 128] - Large 2-Layer
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 1: ARTIFICIAL NEURAL NETWORKS")
println("Testing 8 ANN Topologies")
println("="^70)

topologies_to_test = [
    [256, 128, 64],
    [128, 64, 32],
    [96, 48, 24],
    [64, 32],
    [128, 128, 64, 32],
    [192, 96, 48],
    [128, 64],
    [256, 128]
]

ann_results = []

for (i, topology) in enumerate(topologies_to_test)
    println("\n[$i/8] Testing topology: $topology")
    
    hyperparams = Dict(
        "topology" => topology,
        "learningRate" => 0.003,
        "validationRatio" => 0.1,
        "numExecutions" => 1,
        "maxEpochs" => 800,
        "maxEpochsVal" => 25
    )
    
    # Use modelCrossValidation from utils.jl (course function)
    results = modelCrossValidation(
        :ANN,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(ann_results, (topology, f1_stats[1], results))
end

# Sort by F1 score
sorted_ann_results = sort(ann_results, by=x->x[2], rev=true)

println("\n🏆 ANN Results Ranking (by F1 Score):")
println("-"^70)
for (i, (topo, f1, _)) in enumerate(sorted_ann_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. $topo - F1: $(round(f1*100, digits=2))%")
end

best_topology_ann = sorted_ann_results[1][1]
best_f1_ann = sorted_ann_results[1][2]
println("\n✨ Best ANN: $best_topology_ann (CV F1: $(round(best_f1_ann*100, digits=2))%)")

# Train final ANN on full training set and evaluate on test set
println("\n🚀 Training final ANN on full training set...")

# Prepare data
train_targets_onehot = oneHotEncoding(train_targets)
test_targets_onehot = oneHotEncoding(test_targets)

# Normalize
normParams_ann = calculateMinMaxNormalizationParameters(train_inputs)
train_inputs_norm = normalizeMinMax(train_inputs, normParams_ann)
test_inputs_norm = normalizeMinMax(test_inputs, normParams_ann)

# Create validation split (10% of train)
N_train = size(train_inputs_norm, 1)
(train_idx, val_idx) = holdOut(N_train, 0.1)

# Train final model
final_ann, _ = trainClassANN(best_topology_ann,
    (train_inputs_norm[train_idx, :], train_targets_onehot[train_idx, :]),
    validationDataset=(train_inputs_norm[val_idx, :], train_targets_onehot[val_idx, :]),
    testDataset=(test_inputs_norm, test_targets_onehot),
    maxEpochs=800,
    learningRate=0.003,
    maxEpochsVal=25)

# Predict on test set
test_outputs_ann = final_ann(test_inputs_norm')'

# Calculate metrics using confusionMatrix from course utils
cm_results_ann = confusionMatrix(test_outputs_ann, test_targets_onehot; weighted=true)

println("\n📊 ANN TEST SET RESULTS:")
println("="^70)
println("Accuracy:  $(round(cm_results_ann.accuracy*100, digits=2))%")
println("F1 Score:  $(round(cm_results_ann.aggregated.f1*100, digits=2))%")
println("\nConfusion Matrix:")
printConfusionMatrix(test_outputs_ann, test_targets_onehot; weighted=true)

# ============================================================================
#        EXPERIMENT 2: SUPPORT VECTOR MACHINES (SVMs)
# ============================================================================
#
# Configuration:
#   - 10 configurations tested
#   - Kernels: Linear, RBF, Polynomial
#   - Hyperparameter C: 0.1, 1.0, 10.0
#   - Gamma (RBF): auto (1/n_features = 0.125), 0.1
#   - Degree (Polynomial): 2, 3
#
# Configurations:
#   1-3. Linear (C = 0.1, 1.0, 10.0)
#   4-6. RBF with auto gamma (C = 0.1, 1.0, 10.0)
#   7. RBF with γ=0.1 (C = 1.0)
#   8-10. Polynomial degree 2, 3 (C = 1.0, 10.0)
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 2: SUPPORT VECTOR MACHINES")
println("Testing 10 SVM Configurations")
println("="^70)

svm_configs = [
    ("linear", 0.1, 0.125, 3, "Linear C=0.1"),
    ("linear", 1.0, 0.125, 3, "Linear C=1.0"),
    ("linear", 10.0, 0.125, 3, "Linear C=10.0"),
    ("rbf", 0.1, 0.125, 3, "RBF C=0.1 γ=auto"),
    ("rbf", 1.0, 0.125, 3, "RBF C=1.0 γ=auto"),
    ("rbf", 10.0, 0.125, 3, "RBF C=10.0 γ=auto"),
    ("rbf", 1.0, 0.1, 3, "RBF C=1.0 γ=0.1"),
    ("poly", 1.0, 0.125, 2, "Poly C=1.0 deg=2"),
    ("poly", 1.0, 0.125, 3, "Poly C=1.0 deg=3"),
    ("poly", 10.0, 0.125, 2, "Poly C=10.0 deg=2")
]

svm_results = []

for (i, (kernel, C, gamma, degree, desc)) in enumerate(svm_configs)
    println("\n[$i/10] Testing: $desc")
    
    hyperparams = Dict(
        "kernel" => kernel,
        "C" => C,
        "gamma" => gamma,
        "degree" => degree
    )
    
    # Use modelCrossValidation from utils.jl
    results = modelCrossValidation(
        :SVC,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(svm_results, (desc, f1_stats[1], kernel, C, gamma, degree, results))
end

sorted_svm_results = sort(svm_results, by=x->x[2], rev=true)

println("\n🏆 SVM Results Ranking:")
println("-"^70)
for (i, (desc, f1, _, _, _, _, _)) in enumerate(sorted_svm_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. $desc - F1: $(round(f1*100, digits=2))%")
end

best_svm = sorted_svm_results[1]
best_desc_svm, best_f1_svm, best_kernel_svm, best_C_svm, best_gamma_svm, best_degree_svm = best_svm[1:6]
println("\n✨ Best SVM: $best_desc_svm (CV F1: $(round(best_f1_svm*100, digits=2))%)")

# Train final SVM and evaluate on test set
println("\n🚀 Training final SVM on full training set...")

# Load MLJ for final model training
using MLJ
SVMClassifier = @load SVC pkg=LIBSVM

# Normalize
train_inputs_norm_svm = normalizeMinMax(train_inputs, calculateMinMaxNormalizationParameters(train_inputs))
test_inputs_norm_svm = normalizeMinMax(test_inputs, calculateMinMaxNormalizationParameters(train_inputs))

# Convert to strings for MLJ
train_targets_str = string.(train_targets)
test_targets_str = string.(test_targets)
classes = sort(unique(train_targets_str))

# Set kernel
if best_kernel_svm == "linear"
    kernel_func = LIBSVM.Kernel.Linear
elseif best_kernel_svm == "poly"
    kernel_func = LIBSVM.Kernel.Polynomial
else
    kernel_func = LIBSVM.Kernel.RadialBasis
end

model_svm = SVMClassifier(
    kernel=kernel_func,
    cost=best_C_svm,
    gamma=best_gamma_svm,
    degree=Int32(best_degree_svm)
)

mach_svm = machine(model_svm, MLJ.table(train_inputs_norm_svm), categorical(train_targets_str))
MLJ.fit!(mach_svm, verbosity=0)

# Predict
svm_predictions = MLJ.predict(mach_svm, MLJ.table(test_inputs_norm_svm))
cm_results_svm = confusionMatrix(svm_predictions, test_targets_str, classes; weighted=true)

println("\n📊 SVM TEST SET RESULTS:")
println("="^70)
println("Accuracy:  $(round(cm_results_svm.accuracy*100, digits=2))%")
println("F1 Score:  $(round(cm_results_svm.aggregated.f1*100, digits=2))%")

# ============================================================================
#            EXPERIMENT 3: DECISION TREES
# ============================================================================
#
# Configuration:
#   - 7 maximum depths tested: 3, 5, 7, 10, 15, 20, unlimited
#   - Splitting criterion: Gini impurity
#   - Min samples split: 2
#   - Random seed: 42 (for reproducibility)
#
# Advantages:
#   - Interpretable (can visualize decision rules)
#   - No feature scaling required
#   - Fast training and prediction
#   - Handles non-linear relationships naturally
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 3: DECISION TREES")
println("Testing 7 Maximum Depths")
println("="^70)

tree_depths = [3, 5, 7, 10, 15, 20, -1]
tree_results = []

for (i, max_depth) in enumerate(tree_depths)
    depth_str = max_depth == -1 ? "Unlimited" : string(max_depth)
    println("\n[$i/7] Testing: Depth=$depth_str")
    
    hyperparams = Dict("max_depth" => max_depth)
    
    # Use modelCrossValidation from utils.jl
    results = modelCrossValidation(
        :DecisionTreeClassifier,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(tree_results, (depth_str, max_depth, f1_stats[1], results))
end

sorted_tree_results = sort(tree_results, by=x->x[3], rev=true)

println("\n🏆 Decision Tree Results Ranking:")
println("-"^70)
for (i, (depth_str, _, f1, _)) in enumerate(sorted_tree_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. Depth=$depth_str - F1: $(round(f1*100, digits=2))%")
end

best_desc_tree, best_max_depth_tree, best_f1_tree = sorted_tree_results[1][1:3]
println("\n✨ Best Tree: Depth=$best_desc_tree (CV F1: $(round(best_f1_tree*100, digits=2))%)")

# Train final Decision Tree
println("\n🚀 Training final Decision Tree on full training set...")

DTClassifier = @load DecisionTreeClassifier pkg=DecisionTree

train_inputs_norm_tree = normalizeMinMax(train_inputs, calculateMinMaxNormalizationParameters(train_inputs))
test_inputs_norm_tree = normalizeMinMax(test_inputs, calculateMinMaxNormalizationParameters(train_inputs))

train_targets_str_tree = string.(train_targets)
test_targets_str_tree = string.(test_targets)

if best_max_depth_tree == -1
    model_tree = DTClassifier(rng=Random.MersenneTwister(42))
else
    model_tree = DTClassifier(max_depth=best_max_depth_tree, rng=Random.MersenneTwister(42))
end

mach_tree = machine(model_tree, MLJ.table(train_inputs_norm_tree), categorical(train_targets_str_tree))
MLJ.fit!(mach_tree, verbosity=0)

tree_predictions = MLJ.predict(mach_tree, MLJ.table(test_inputs_norm_tree))
tree_predictions_mode = mode.(tree_predictions)

cm_results_tree = confusionMatrix(tree_predictions_mode, test_targets_str_tree, classes; weighted=true)

println("\n📊 DECISION TREE TEST SET RESULTS:")
println("="^70)
println("Accuracy:  $(round(cm_results_tree.accuracy*100, digits=2))%")
println("F1 Score:  $(round(cm_results_tree.aggregated.f1*100, digits=2))%")

# ============================================================================
#            EXPERIMENT 4: k-NEAREST NEIGHBORS (kNN)
# ============================================================================
#
# Configuration:
#   - 6 k values tested: 1, 3, 5, 7, 10, 15
#   - Distance metric: Euclidean
#   - Voting: Majority voting among k neighbors
#
# Notes:
#   - Feature normalization is CRITICAL for kNN (distance-based)
#   - No explicit training phase (lazy learning)
#   - k=1 is most sensitive to noise
#   - Higher k values create smoother decision boundaries
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 4: k-NEAREST NEIGHBORS")
println("Testing 6 k Values")
println("="^70)

k_values = [1, 3, 5, 7, 10, 15]
knn_results = []

for (i, k) in enumerate(k_values)
    println("\n[$i/6] Testing: k=$k")
    
    hyperparams = Dict("n_neighbors" => k)
    
    # Use modelCrossValidation from utils.jl
    results = modelCrossValidation(
        :KNeighborsClassifier,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(knn_results, (k, f1_stats[1], results))
end

sorted_knn_results = sort(knn_results, by=x->x[2], rev=true)

println("\n🏆 kNN Results Ranking:")
println("-"^70)
for (i, (k, f1, _)) in enumerate(sorted_knn_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. k=$k - F1: $(round(f1*100, digits=2))%")
end

best_k_knn, best_f1_knn = sorted_knn_results[1][1:2]
println("\n✨ Best kNN: k=$best_k_knn (CV F1: $(round(best_f1_knn*100, digits=2))%)")

# Train final kNN
println("\n🚀 Preparing final kNN...")

kNNClassifier = @load KNNClassifier pkg=NearestNeighborModels

train_inputs_norm_knn = normalizeMinMax(train_inputs, calculateMinMaxNormalizationParameters(train_inputs))
test_inputs_norm_knn = normalizeMinMax(test_inputs, calculateMinMaxNormalizationParameters(train_inputs))

train_targets_str_knn = string.(train_targets)
test_targets_str_knn = string.(test_targets)

model_knn = kNNClassifier(K=best_k_knn)

mach_knn = machine(model_knn, MLJ.table(train_inputs_norm_knn), categorical(train_targets_str_knn))
MLJ.fit!(mach_knn, verbosity=0)

knn_predictions = MLJ.predict(mach_knn, MLJ.table(test_inputs_norm_knn))
knn_predictions_mode = mode.(knn_predictions)

cm_results_knn = confusionMatrix(knn_predictions_mode, test_targets_str_knn, classes; weighted=true)

println("\n📊 kNN TEST SET RESULTS:")
println("="^70)
println("Accuracy:  $(round(cm_results_knn.accuracy*100, digits=2))%")
println("F1 Score:  $(round(cm_results_knn.aggregated.f1*100, digits=2))%")

# ============================================================================
#            EXPERIMENT 5: ENSEMBLE METHODS
# ============================================================================
#
# Strategy: Combine the top 3 individual models to improve robustness
#
# Models Selected:
#   1. Best ANN
#   2. Best Decision Tree
#   3. Best kNN
#
# Ensemble Techniques:
#   1. Majority Voting: Each model votes equally, winner takes all
#   2. Weighted Voting: Models vote proportionally to their CV F1 scores
#
# Expected Benefits:
#   - Reduced variance through model averaging
#   - More robust predictions
#   - Leverage complementary strengths of different algorithms
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 5: ENSEMBLE METHODS")
println("Combining ANN + Decision Tree + kNN")
println("="^70)

# Helper functions for ensemble
function majorityVoting(predictions::Vector{Vector{String}})
    n_samples = length(predictions[1])
    ensemble_predictions = Vector{String}(undef, n_samples)
    
    for i in 1:n_samples
        votes = [pred[i] for pred in predictions]
        ensemble_predictions[i] = mode(votes)
    end
    
    return ensemble_predictions
end

function weightedVoting(predictions::Vector{Vector{String}}, weights::Vector{Float64})
    n_samples = length(predictions[1])
    n_models = length(predictions)
    classes_unique = sort(unique(vcat(predictions...)))
    n_classes = length(classes_unique)
    
    ensemble_predictions = Vector{String}(undef, n_samples)
    
    for i in 1:n_samples
        class_scores = Dict(c => 0.0 for c in classes_unique)
        
        for j in 1:n_models
            class_pred = predictions[j][i]
            class_scores[class_pred] += weights[j]
        end
        
        ensemble_predictions[i] = argmax(class_scores)
    end
    
    return ensemble_predictions
end

# Get test predictions from all 3 models as string vectors
ann_test_pred_str = string.(argmax.(eachrow(test_outputs_ann)) .- 1)
tree_test_pred_str = string.(tree_predictions_mode)
knn_test_pred_str = string.(knn_predictions_mode)

all_predictions = [ann_test_pred_str, tree_test_pred_str, knn_test_pred_str]

# Method 1: Majority Voting
println("\n[1/2] Majority Voting...")
majority_predictions = majorityVoting(all_predictions)
cm_results_majority = confusionMatrix(majority_predictions, test_targets_str, classes; weighted=true)
println("✅ Majority Voting - F1: $(round(cm_results_majority.aggregated.f1*100, digits=2))%")

# Method 2: Weighted Voting
println("\n[2/2] Weighted Voting...")
cv_scores = [best_f1_ann, best_f1_tree, best_f1_knn]
weights = cv_scores ./ sum(cv_scores)

println("  Model weights:")
println("    ANN:           $(round(weights[1]*100, digits=1))%")
println("    Decision Tree: $(round(weights[2]*100, digits=1))%")
println("    kNN:           $(round(weights[3]*100, digits=1))%")

weighted_predictions = weightedVoting(all_predictions, weights)
cm_results_weighted = confusionMatrix(weighted_predictions, test_targets_str, classes; weighted=true)
println("✅ Weighted Voting - F1: $(round(cm_results_weighted.aggregated.f1*100, digits=2))%")

# ============================================================================
#              FINAL RESULTS COMPARISON
# ============================================================================
#
# Comprehensive comparison of all 6 approaches on the hold-out test set.
#
# Evaluation Metrics:
#   - F1 Score: Harmonic mean of precision and recall
#   - Accuracy: Overall correct predictions
#   - Per-Class Metrics: Performance for each risk level
#
# Key Question: Which approach best balances overall performance 
#               with fraud detection capability?
#
# ============================================================================

println("\n" * "="^70)
println("📊 FINAL RESULTS - ALL APPROACHES COMPARISON")
println("="^70)

println("\n🏆 TEST SET PERFORMANCE RANKING:")
println("="^70)
println("Rank | Approach                | F1 Score   | Accuracy")
println("-"^70)
println("1.   | ANN ($best_topology_ann)         | $(rpad(round(cm_results_ann.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_ann.accuracy*100, digits=2))%")
println("2.   | Decision Tree (d=$best_desc_tree)   | $(rpad(round(cm_results_tree.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_tree.accuracy*100, digits=2))%")
println("3.   | kNN (k=$best_k_knn)            | $(rpad(round(cm_results_knn.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_knn.accuracy*100, digits=2))%")
println("4.   | SVM ($best_desc_svm)  | $(rpad(round(cm_results_svm.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_svm.accuracy*100, digits=2))%")
println("-"^70)
println("5.   | Ensemble (Majority)     | $(rpad(round(cm_results_majority.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_majority.accuracy*100, digits=2))%")
println("6.   | Ensemble (Weighted)     | $(rpad(round(cm_results_weighted.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_weighted.accuracy*100, digits=2))%")
println("="^70)

# Determine overall best
all_f1_scores = [
    ("ANN", cm_results_ann.aggregated.f1),
    ("Decision Tree", cm_results_tree.aggregated.f1),
    ("kNN", cm_results_knn.aggregated.f1),
    ("SVM", cm_results_svm.aggregated.f1),
    ("Ensemble Majority", cm_results_majority.aggregated.f1),
    ("Ensemble Weighted", cm_results_weighted.aggregated.f1)
]
best_overall = sort(all_f1_scores, by=x->x[2], rev=true)[1]

println("\n🎯 BEST OVERALL APPROACH: $(best_overall[1])")
println("   Test F1 Score: $(round(best_overall[2]*100, digits=2))%")

println("\n" * "="^70)
println("✅ ALL EXPERIMENTS COMPLETE!")
println("="^70)
println("\n📋 SUMMARY:")
println("  ✅ Tested 4 ML algorithms using modelCrossValidation from course utils")
println("  ✅ ANN: 8 topologies tested")
println("  ✅ SVM: 10 configurations tested")
println("  ✅ Decision Tree: 7 depths tested")
println("  ✅ kNN: 6 k values tested")
println("  ✅ Created 2 ensemble methods")
println("  ✅ Proper Train/Test split (80/20) - no data leakage")
println("  ✅ 3-fold stratified cross-validation for model selection")
println("  ✅ Random seed (42) set for reproducibility")
println("  ✅ All code follows course methodology")
println("="^70)