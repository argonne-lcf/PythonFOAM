// Copyright (C) 2021 UChicago Argonne, LLC
// SPDX-License-Identifier: (other)
//
// Python-OpenFOAM integration
//
// Send/Receive data with Python

if (runTime.outputTime())
{
    clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX

    // Call APMOS function
    rank_val = PyLong_FromLong(rank);
    PyTuple_SetItem(apmos_args, 0, rank_val);
    pValue = reinterpret_cast<PyArrayObject*>
    (
        PyObject_CallObject(apmos_func, apmos_args)
    );

    clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
    posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
    printf("APMOS Computation wall time: %.2f ms\n", posix_wall);

    clock_gettime(CLOCK_MONOTONIC, &tw1); // C11; use clock_gettime in POSIX

    // Shove return value into OF data structure
    // ndarray return of POD modes
    // double* c_out = static_cast<double*>(PyArray_DATA(pValue));

    ///int shift = truncation*num_cells;
    for (int mode = 0; mode < truncation; ++mode)
    {
        // Overwrite data
        forAll(upod_.internalField(), id) // for boundary field use u_.boundaryField()
        {
            // upod_[id] = (*(c_out+truncation*id+mode));
            // vpod_[id] = (*(c_out+truncation*id+mode+shift));
            // wpod_[id] = (*(c_out+truncation*id+mode+2*shift));
            upod_[id] = *((double*)PyArray_GETPTR2(pValue, id, mode));
            vpod_[id] = *((double*)PyArray_GETPTR2(pValue, num_cells + id, mode));
            wpod_[id] = *((double*)PyArray_GETPTR2(pValue, 2*num_cells + id, mode));
        }

        // Write out
        char var_name[20];
        sprintf(var_name, "upod_%d", mode);
        upod_.rename(&var_name[0]);
        upod_.write();

        sprintf(var_name, "vpod_%d", mode);
        vpod_.rename(&var_name[0]);
        vpod_.write();

        sprintf(var_name, "wpod_%d", mode);
        wpod_.rename(&var_name[0]);
        wpod_.write();

    }

    // std::cout <<"Data back in OpenFOAM. Now writing." << std::endl;

    // delete[] c_out;

    clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
    posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
    printf("Data copy back to OF wall time: %.2f ms\n", posix_wall);

}
else
{
    clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX

    // #pragma omp parallel for
    forAll(upod_.internalField(), id) // for boundary field use u_.boundaryField()
    {
        input_vals[id][0] = U[id].x();
        input_vals[id][1] = U[id].y();
        input_vals[id][2] = U[id].z();
    }

    // Numpy array dimensions
    npy_intp dim[] = {num_cells, 3};
    // create a new array using 'buffer'
    array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);

    clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
    posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
    printf("Numpy array cast time: %.2f ms\n", posix_wall);

    clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX

    // Usage
    getrusage(0,&usage_info);
    std::cout << "Resource usage line 99:" << usage_info.ru_maxrss << std::endl;

    rank_val = PyLong_FromLong(rank);
    PyTuple_SetItem(snapshot_args, 0, array_2d);
    PyTuple_SetItem(snapshot_args, 1, rank_val);
    (void) PyObject_CallObject(snapshot_func, snapshot_args);

    // Usage
    getrusage(0,&usage_info);
    std::cout << "Resource usage line 107:" << usage_info.ru_maxrss << std::endl;

    clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
    posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
    printf("Snapshot collection wall time: %.2f ms\n", posix_wall);
}

// ************************************************************************* //
