echo "Make sure your paths to OpenFOAM/Python headers/libraries are set appropriately"
if [ "$1" == "-debug" ]; then
	echo ""
	echo "Using OpenFOAM in Debug mode"
	source ./OF8_debug/OpenFOAM-8/etc/bashrc
else
	echo "Using OpenFOAM in Optimized mode"
	source /opt/openfoam8/etc/bashrc
fi

export PYTHON_LIB_PATH=/home/rmlans/Desktop/Research/PythonFOAM_Git/ofenv/lib
export PYTHON_BIN_PATH=/home/rmlans/Desktop/Research/PythonFOAM_Git/ofenv/bin
export PYTHON_INCLUDE_PATH=/home/rmlans/Desktop/Research/PythonFOAM_Git/ofenv/include/python3.6m
export NUMPY_INCLUDE_PATH=/home/rmlans/Desktop/Research/PythonFOAM_Git/ofenv/lib/python3.6/site-packages/numpy/core/include
export PYTHON_LIB_NAME=lpython3.6m


export LD_LIBRARY_PATH=$PYTHON_LIB_PATH:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PYTHON_LIB_PATH:$LIBRARY_PATH
export PATH=$PYTHON_BIN_PATH:$PATH

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rmlans/Desktop/ROMS/TensorFlowFoam/tf_c_api/lib
# export LIBRARY_PATH=$LIBRARY_PATH:/home/rmlans/Desktop/ROMS/TensorFlowFoam/tf_c_api/lib