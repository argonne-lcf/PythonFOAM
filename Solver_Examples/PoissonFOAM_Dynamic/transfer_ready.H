
// Velocity field for transfer to Python
 forAll(U.internalField(), id)  
        {
        input_vals[id][0] = U[id].component(vector::X);
        input_vals[id][1] = U[id].component(vector::Y);
        }
npy_intp dim[] = {num_cells,2};
array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);
PyTuple_SetItem(nn_model_args, 0, array_2d);
    
    
//////////////////////////////////////////////////////// / 
// A matrix for transfer to python   
    
for(label i=0; i<num_cells; i++)
	{
	input_mat[i][i] = pEqn.diag()[i];
	}
for(label faceI=0;faceI<pEqn.lduAddr().lowerAddr().size(); faceI++)
	{
	label l = pEqn.lduAddr().lowerAddr()[faceI];
	label u = pEqn.lduAddr().upperAddr()[faceI];
	input_mat[l][u] = pEqn.upper()[faceI];
	input_mat[u][l] = pEqn.upper()[faceI];
	}
	// Assigning contribution from BC

forAll(p.boundaryField(), patchI)
	{
	const fvPatch &p_1 =
	p.boundaryField()[patchI].patch();
	forAll(p_1, faceI)
	{	
	label cellI = p_1.faceCells()[faceI];
	input_mat[cellI][cellI]
	+= pEqn.internalCoeffs()[patchI][faceI];
	}
	}
npy_intp dim_2[] = {num_cells,num_cells};
array_2d_2 = PyArray_SimpleNewFromData(2, dim_2, NPY_DOUBLE, &input_mat[0]);
PyTuple_SetItem(nn_model_args, 1, array_2d_2);
 
 /////////////////////////////////////////////////////////////////////////////////   
// B Matrix for transfer to Python
npy_intp dim_3[] = {num_cells};
forAll(div_phiHbyA.internalField(), id) 
        {
        input_b[id] = div_phiHbyA[id];
        }
array_2d_3 = PyArray_SimpleNewFromData(1, dim_3, NPY_DOUBLE, &input_b[0]);
PyTuple_SetItem(nn_model_args, 2, array_2d_3);


///////////////////////////////////////////////////////////////////
// Cell volumes to be transferred to Python
npy_intp dim_4[] = {num_cells};
forAll(mesh.V(), id) 
        {
        input_mv[id] = mesh.V()[id];
        }
array_2d_4 = PyArray_SimpleNewFromData(1, dim_4, NPY_DOUBLE, &input_mv[0]);
PyTuple_SetItem(nn_model_args, 3, array_2d_4);


///////////////////////////////////////////////////////////////////
// Train mode or test mode
PyObject* py_int = PyLong_FromLong(train_mode);
PyTuple_SetItem(nn_model_args, 4, py_int);
Py_DECREF(py_int);


///////////////////////////////////////////////////////////////////////////////////
pValue = (PyArrayObject*)PyObject_CallObject(nn_model, nn_model_args);
forAll(p.internalField(), id) 
        {
        p[id] = *((double*)PyArray_GETPTR1(pValue, id));
        }
        



    

    
