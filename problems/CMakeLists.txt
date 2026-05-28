set(PROBLEMS
    kelvin_helmholtz
    sod_shock
    rayleigh_taylor
    heat_gaussian
    isentropic_vortex
)

foreach(problem IN LISTS PROBLEMS)
    add_subdirectory(${problem})
endforeach()
