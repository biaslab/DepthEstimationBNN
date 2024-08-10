using ReTestItems, CpuId, Coverage
using UnboundedBNN

cpu = cpucores()

runtests(
    UnboundedBNN,
    nworkers = cpu == 0 ? 1 : cpu,
    nworker_threads = cpu == 0 ? 1 : Int(cputhreads() / cpucores()),
    memory_threshold = 1.0
)

# Generate coverage report for CI.yml
if get(ENV, "COVERAGE", "false") == "true"
    @info "Generating coverage report"
    coverage = process_folder(joinpath(pkgdir(UnboundedBNN), "src"))
    LCOV.writefile(joinpath(pkgdir(UnboundedBNN), "lcov.info"), coverage)
end