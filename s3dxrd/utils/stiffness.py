import numpy as np
from numpy import ndarray
from xfab import tools

def alpha_quartz_stiffness(isotropy):
    """
    Returns the stiffness matrix for α-silicon dioxide, as defined in Voigt notation.

    :returns: The 6x6 stiffness matrix for α-silicon dioxide as a numpy array.
    :rtype: ndarray

    """
    if isotropy:
        ny = 0.17
        E = 70*1e9
        diag = np.array([1-ny, 1-ny, 1-ny, 0.5*(1-2*ny), 0.5*(1-2*ny), 0.5*(1-2*ny)])
        C = np.diag(diag, 0)
        C[0,1] = C[0,2] = C[1,2] = C[1,0] = C[2,0] = C[2,1] = ny
        return C*(E/((1+ny)*(1-2*ny)))
    
    C11, C33, C44, C66, C12, C13, C14 = 86.99, 106.39, 58.12, 40.12, 6.75, 12.17, 17.99

    C = np.zeros((6, 6))
    C[0, :4] = [C11, C12, C13, C14]
    C[1, 1:4] = [C11, C13, -C14]
    C[2, 2] = C33
    C[3, 3] = C44
    C[4, 4:6] = [C44, C14]
    C[5, 5] = C66

    C = C + np.tril(C.T, -1)
    C = C * (10 ** 9)  # Convert to GPa
    return np.array(C)


def transform_stiffness(U, C):
    """Transform a stiffness matrix defined in the grain coordinate system into a stiffness matrix defined in the
    sample coordinate system by applying a rotational matrix defined from the grain orientation matrix U.

    :param U:  Orientation/rotation tensor as a 3x3 numpy array.
    :type U: ndarray
    :param C: Stiffness matrix as a 6x6 numpy array.
    :type C: ndarray
    :returns: A stiffness matrix valid in the sample coordinate system as a 6x6 numpy array.
    :rtype: ndarray

    """
    M = _get_rotation_matrix(U)
    C_rot = (M @ C @ np.transpose(M))

    return C_rot


def _get_rotation_matrix(U):
    """
    Private function for creating the rotational matrices used in :func:`transform_stiffness`. For further information
    see B.A Auld, *Acoustic fields and waves in solids*.

    :param U: Orientation/rotation tensor as a 3x3 numpy array.
    :type U: ndarray
    :return: Rotational matrix as a 6x6 numpy array.
    :rtype: ndarray
    """

    M = np.array([[U[0, 0] ** 2, U[0, 1] ** 2, U[0, 2] ** 2, 2 * U[0, 1] * U[0, 2], 2 * U[0, 2] * U[0, 0],
                   2 * U[0, 0] * U[0, 1]],
                  [U[1, 0] ** 2, U[1, 1] ** 2, U[1, 2] ** 2, 2 * U[1, 1] * U[1, 2], 2 * U[1, 2] * U[1, 0],
                   2 * U[1, 0] * U[1, 1]],
                  [U[2, 0] ** 2, U[2, 1] ** 2, U[2, 2] ** 2, 2 * U[2, 1] * U[2, 2], 2 * U[2, 2] * U[2, 0],
                   2 * U[2, 0] * U[2, 1]],

                  [U[1, 0] * U[2, 0], U[1, 1] * U[2, 1], U[1, 2] * U[2, 2], U[1, 1] * U[2, 2] + U[1, 2] * U[2, 1],
                   U[1, 0] * U[2, 2] + U[1, 2] * U[2, 0], U[1, 1] * U[2, 0] + U[1, 0] * U[2, 1]],
                  [U[2, 0] * U[0, 0], U[2, 1] * U[0, 1], U[2, 2] * U[0, 2], U[0, 1] * U[2, 2] + U[0, 2] * U[2, 1],
                   U[0, 2] * U[2, 0] + U[0, 0] * U[2, 2], U[0, 0] * U[2, 1] + U[0, 1] * U[2, 0]],
                  [U[0, 0] * U[1, 0], U[0, 1] * U[1, 1], U[0, 2] * U[1, 2], U[0, 1] * U[1, 2] + U[0, 2] * U[1, 1],
                   U[0, 2] * U[1, 0] + U[0, 0] * U[1, 2], U[0, 0] * U[1, 1] + U[0, 1] * U[1, 0]]])
    return M


def calculate_stress_by_matrix_rotation(wlsq_strain, U, isotropy=False):
    """
    Calculate the intra-granular stresses by applying the sample system stiffnes matrix for a given grain and z-slice
    to the corresponding strains.

    :param wlsq_strain: Strains as a list of numpy arrays, where each list contains a strain component. The order of
        the strains are ["XX", "YY", "ZZ", "YZ", "XZ", "XY"].
    :type wlsq_strain: list[ndarray]
    :param U: The orientation matrix for a given grain and z-slice.
    :type U: ndarray
    :param isotropy: Toggle material isotropy on/off.
    :type isotropy: bool
    :return: The intra-granular stresses in the same format as the provided intragranular strains.
    :rtype: list[ndarray]
    """
    # Get the stiffness matrix as measured in the grain coordinate system
    C = alpha_quartz_stiffness(isotropy)
    # Account for the discrepancy between the IEEE definition of the coordinate system, used for the stiffness matrix,
    # and Poulsens definition of the grain coordinate system by rotating the vector to the IEEE system.
    E = get_alignment_mat()
    # Rotate the stiffness matrix by the grain orientation matrix
    C = transform_stiffness(U @ E, C)
    # Stack the strain vectors into a matrix, where each row contains the strain components for a certain element in
    # the mesh which the stress will be plotted on. Make an empty matrix for the stress vectors.
    strain_mat = np.column_stack(wlsq_strain)
    stress_mat = np.zeros_like(strain_mat)

    # Exract a row from the strain matrix, multiply the shear strain components by 2 to obtain the engineeering shear
    # strain which is compatible with the Voigt notation.
    for i in range(np.size(strain_mat, 0)):
        strain_vector = strain_mat[i, :]
        strain_vector[3:6] *= 2
        # Apply the stiffness matrix to get the stress vectors and stack the stress vectors in a matrix.
        stress_mat[i, :] = C @ strain_vector
    # Split the stress matrix to give it the same format as wlsq_strains.
    wlsq_stress = np.hsplit(stress_mat, 6)

    for i, arr in enumerate(wlsq_stress):
        wlsq_stress[i] = arr.reshape((-1))

    return wlsq_stress


def calculate_stress_by_vector_rotation(wlsq_strain, U, isotropy=False):
    """
       Calculate the intra-granular stresses by converting the strains into a 3x3 tensor format and using the grain
       orientation matrix to calculate the strains in the grain coordiante system. The stiffness matrix is then applied
       in the grain coordinate system and the resulting stresses are transformed back to the sample coordiante system.

       :param wlsq_strain: Strains as a list of numpy arrays, where each list contains a strain component. The order of
           the strains are ["XX", "YY", "ZZ", "YZ", "XZ", "XY"].
       :type wlsq_strain: list[ndarray]
       :param U: The orientation matrix for a given grain and z-slice.
       :type U: ndarray
       :param isotropy: Toggle material isotropy on/off.
       :type isotropy: bool
       :return: The intra-granular stresses in the same format as the provided intragranular strains.
       :rtype: list[ndarray]
       """
    # Here the stresses will be calculated using a different method where the strain vectors are rotated into the grain
    # coordinate system where the stiffness matrix is applied and then the corresponding strain vector is rotated back
    # by solving a system of equations.

    # Get the stiffness matrix as measured in the grain coordinate system
    C = alpha_quartz_stiffness(isotropy)

    # Stack the strain vectors into a matrix, where each row contains the strain components for a certain element in
    # the mesh which the stress will be plotted on. Make an empty matrix for the stress vectors.
    strain_mat = np.column_stack(wlsq_strain)
    stress_mat = np.zeros_like(strain_mat)

    # Account for the discrepancy between the IEEE definition of the coordinate system, used for the stiffness matrix,
    # and Poulsens definition of the grain coordinate system by rotating the vector to the IEEE system.
    E = get_alignment_mat()

    for i in range(np.size(strain_mat, 0)):
        strain_vector = strain_mat[i, :]
        # Transform the strain_vector to the grain coordinate system.
        strain_tensor = vec_to_tens(strain_vector)
        grain_strain_tensor = E.T @ (U.T @ strain_tensor @ U) @ E

        # Convert the grain strain vector to Voigt notation.
        grain_strain_vector = tens_to_vec(grain_strain_tensor)
        grain_strain_vector[3:6] *= 2

        # Calculate the stress in the grain coordinate system and apply the U matrix to transform the strain back to
        # the sample coordinate system.
        grain_stress_vector = C @ grain_strain_vector

        # Convert the stress vector to the sample coordinate system.
        grain_stess_tensor = vec_to_tens(grain_stress_vector)
        sample_stress_tensor = U @ (E @ grain_stess_tensor @ E.T) @ U.T
        sample_stress_vector = tens_to_vec(sample_stress_tensor)

        stress_mat[i, :] = sample_stress_vector

    # Split the stress matrix to give it the same format as wlsq_strains.
    wlsq_stress = np.hsplit(stress_mat, 6)
    for i, arr in enumerate(wlsq_stress):
        wlsq_stress[i] = arr.reshape((-1))

    return wlsq_stress


def calc_principal(values):
    """
    Calculate the principal values by solving the eigenvalue problem for each reconstructed strain tensor.
    The result is a list of numpy arrays where each numpy array contains one of the principal components.

    :param values: Stresses or strains as a list of numpy arrays, where each list contains a stress or strain component.
    The order of the values should be ["XX", "YY", "ZZ", "YZ", "XZ", "XY"].
    :type values: list[ndarray]

    :return: The principal stresses or strains as a list of numpy arrays. The order of the principal stresses is
        ["X_1", "X_2", "X_3"], where X_1 corresponds to the largest principal value and X_3 corresponds to the smallest
        principal value.
    :rtype: list[ndarray]
    """
    stress = np.column_stack(values)
    nrows = np.size(stress, 0)
    principal_values = np.zeros((nrows, 3))

    for i in range(nrows):
        sigma = vec_to_tens(stress[i, :])
        eigenvals, eigenvects = np.linalg.eig(sigma)
        eigenvals = np.sort(eigenvals)[::-1]  # Should reverse the array so that it is ordered from greatest to least.

        principal_values[i, 0] = eigenvals[0]
        principal_values[i, 1] = eigenvals[1]#sigma[0, 0] + sigma[1, 1] + sigma[2, 2] - eigenvals[0] - eigenvals[2]
        principal_values[i, 2] = eigenvals[2]

    principal_values = np.hsplit(principal_values, 3)
    for i, arr in enumerate(principal_values):
        principal_values[i] = arr.reshape((-1))

    return principal_values


def vec_to_tens(vec):
    """
    Private function for converting 6 component strain or stress vectors into 3x3 tensors. Does not account for
    Voigt notation, i.e do **not** supply a vector where the shear components are multiplied by a factor of two.
    :param vec: 6x1 vector with stress or strain components on the format ["XX", "YY", "ZZ", "YZ", "XZ", "XY"].
    :type vec: ndarray
    :return: The supplied stresses or strains on a 3x3 tensor format.
    :rtype: ndarray
    """
    tens = np.row_stack(np.array([[vec[0], vec[5], vec[4]], [vec[5], vec[1], vec[3]], [vec[4], vec[3], vec[2]]]))
    return tens


def tens_to_vec(tens):
    """
       Private function for converting 3x3 tensors into 6 component strain or stress vectors. Does not account for
       Voigt notation, i.e it will **not** return a vector where the shear components are multiplied by a factor of two.
       :param tens: 3x3 tensor with stress or strain components.
       :type tens: ndarray
       :return: The supplied stresses or strains on a 6x1 vector format as ["XX", "YY", "ZZ", "YZ", "XZ", "XY"].
       :rtype: ndarray
       """
    vec = np.array([tens[0, 0], tens[1, 1], tens[2, 2], tens[1, 2], tens[0, 2], tens[0, 1]])
    return vec

def get_alignment_mat():
    cell = [4.926, 4.926, 5.4189, 90.0, 90.0, 120.0]  # From fit.par
    B = tools.form_b_mat(cell)
    A = np.linalg.inv((1 / (2 * np.pi)) * B.T)
    E = np.column_stack((A[:, 0], np.cross(A[:, 2], A[:, 0]), A[:, 2]))
    # Normalize the new vectors
    for i in range(3):
        E[:, i] /= np.linalg.norm(E[:, i])

    return E