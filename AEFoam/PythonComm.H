/* Trigger an autoencoder training using tensorflow in autoencoder_func*/
if (runTime.outputTime())
{
    if (encode_mode == 1)
    {
        //Print out the reconstructions obtained from python
        printf("Storing out reconstructions\n");

        // Write out
        char var_name[20];
        sprintf(var_name, "urec_");
        urec_.rename(&var_name[0]);
        urec_.write();

        encode_mode = 0;
    }
    else
    {
        // To pass rank to Python interpreter
        rank_val = PyLong_FromLong(rank);
        PyTuple_SetItem(autoencoder_args, 0, rank_val);

        // Call python function
        clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX
        
        pValue = (PyArrayObject*)PyObject_CallObject(autoencoder_func, autoencoder_args); //Calling Autoencoder training function
        
        clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
        posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
        printf("Autoencoder training wall time: %.2f ms\n", posix_wall);        

        encode_mode = 1;
    }

}
else
{
    clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX

    // #pragma omp parallel for
    forAll(upod_.internalField(), id) // for boundary field use u_.boundaryField()
    {
        input_vals[id][0] = U[id].x();
    }

    // Numpy array dimensions
    npy_intp dim[] = {num_cells, 1};
    // create a new array using 'buffer'
    array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);
    
    clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
    posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
    printf("Numpy array cast time: %.2f ms\n", posix_wall);
    
    if (encode_mode == 1)
    {
        PyTuple_SetItem(encode_args, 0, array_2d);
        // To pass rank to Python interpreter
        rank_val = PyLong_FromLong(rank);
        PyTuple_SetItem(encode_args, 1, rank_val);

        // Call python function
        clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX

        pValue = (PyArrayObject*)PyObject_CallObject(encode_func, encode_args); //Casting to PyArrayObject

        clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
        posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
        printf("Encoder wall time: %.2f ms\n", posix_wall);

        // Send data back to OpenFOAM of reconstruction
        forAll(urec_.internalField(), id)
        {
            urec_[id] = *((double*)PyArray_GETPTR2(pValue, id, 0));
        }
    }
    else
    {      
        PyTuple_SetItem(snapshot_args, 0, array_2d);
        // To pass rank to Python interpreter
        rank_val = PyLong_FromLong(rank);
        PyTuple_SetItem(snapshot_args, 1, rank_val);

        // Call python function
        clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX

        pValue = (PyArrayObject*)PyObject_CallObject(snapshot_func, snapshot_args); //Casting to PyArrayObject

        clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
        posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
        printf("Snapshot collection wall time: %.2f ms\n", posix_wall);

    }
}
