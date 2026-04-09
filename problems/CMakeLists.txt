set(PROBLEMS
    kelvin_helmholtz
    sod_shock
    rayleigh_taylor
    heat_gaussian
)

foreach(problem IN LISTS PROBLEMS)
    add_subdirectory(${problem})
endforeach()
