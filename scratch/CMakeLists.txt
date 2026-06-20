build_exec(
    EXECNAME slice-rl-sim
    SOURCE_FILES
        slice-rl-sim.cc
        slice-env.cc
    LIBRARIES_TO_LINK
        ${libcore}
        ${libnetwork}
        ${libinternet}
        ${libmobility}
        ${libapplications}
        ${libpoint-to-point}
        ${libflow-monitor}
        ${libnr}
        ${libopengym}
    EXECUTABLE_DIRECTORY_PATH
        scratch/slice-rl/
)

build_exec(
    EXECNAME slice-rl-sim-baseline
    SOURCE_FILES
        slice-rl-sim-baseline.cc
        slice-baseline-env.cc
    LIBRARIES_TO_LINK
        ${libcore}
        ${libnetwork}
        ${libinternet}
        ${libmobility}
        ${libapplications}
        ${libpoint-to-point}
        ${libflow-monitor}
        ${libnr}
        ${libopengym}
    EXECUTABLE_DIRECTORY_PATH
        scratch/slice-rl/
)
