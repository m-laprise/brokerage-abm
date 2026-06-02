using Test

@testset "TransientBrokerage v0.2" begin
    include("test_types.jl")
    include("test_matching_function.jl")
    include("test_learning.jl")
    include("test_network.jl")
    include("test_search.jl")
    include("test_matching.jl")
    include("test_capture.jl")
    include("test_pseudocode_conformance.jl")
    include("test_entry_exit.jl")
    include("test_initialization.jl")
    include("test_measures.jl")
    include("test_diagnostics.jl")
    include("test_invariants.jl")
    include("test_step.jl")
    include("test_integration.jl")
    include("test_regression_baseline.jl")
end
