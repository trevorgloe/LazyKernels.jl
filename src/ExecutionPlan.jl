# File containing the definitions for LazyKernels execution plans
import KernelAbstractions

struct ExecutionPlan
    backend::KernelAbstractions.Backend
    workgroup_n::Int
end
