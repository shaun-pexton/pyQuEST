import pyquest.core

cdef class Damping(SingleQubitOperator):
    def __cinit__(self, target, prob):
        self.TYPE = OP_TYPES.OP_DAMP
        self._prob = prob

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.mixDamping(c_register, self._target, self._prob)


cdef class Dephasing(MultiQubitOperator):
    def __cinit__(self, targets, prob):
        self.TYPE = OP_TYPES.OP_DEPHASE
        if not 0 < self._num_targets < 3:
            raise ValueError("Dephasing noise must act on 1 or 2 qubits.")
        self._prob = prob

    @property
    def prob(self):
        return self._prob

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_targets == 1:
            quest.mixDephasing(c_register, self._targets[0], self._prob)
        # Check for number of targets as extra safeguard.
        elif self._num_targets == 2:
            quest.mixTwoQubitDephasing(
                c_register, self._targets[0], self._targets[1], self._prob)

    def __repr__(self):
        return type(self).__name__ + "(" + str(self.targets) + ", " + str(self._prob) + ")"


cdef class Depolarising(MultiQubitOperator):
    def __cinit__(self, target, prob):
        self.TYPE = OP_TYPES.OP_DEPOL
        if not 0 < self._num_targets < 3:
            raise ValueError("Depolarising noise must act on 1 or 2 qubits.")
        self._prob = prob

    @property
    def prob(self):
        return self._prob

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_targets == 1:
            quest.mixDepolarising(c_register, self._targets[0], self._prob)
        if self._num_targets == 2:
            quest.mixTwoQubitDepolarising(
                c_register, self._targets[0], self._targets[1], self._prob)

    def __repr__(self):
        return type(self).__name__ + "(" + str(self.targets) + ", " + str(self._prob) + ")"


cdef class KrausMap(MultiQubitOperator):

    def __cinit__(self, targets=None, operators=None, target=None):
        self.TYPE = OP_TYPES.OP_KRAUS
        cdef size_t mat_dim = 1 << self._num_targets
        cdef size_t k, n, m
        cdef np.ndarray cur_op
        self._num_ops = len(operators)
        if self._num_targets == 1:
            self._ops = malloc(self._num_ops * sizeof(ComplexMatrix2))
            for m in range(self._num_ops):
                cur_op = operators[m]
                for k in range(mat_dim):
                    for n in range(mat_dim):
                        (<ComplexMatrix2*>self._ops)[m].real[n][k] = cur_op[n, k].real
                        (<ComplexMatrix2*>self._ops)[m].imag[n][k] = cur_op[n, k].imag
        elif self._num_targets == 2:
            self._ops = malloc(self._num_ops * sizeof(ComplexMatrix4))
            for m in range(self._num_ops):
                cur_op = operators[m]
                for k in range(mat_dim):
                    for n in range(mat_dim):
                        (<ComplexMatrix4*>self._ops)[m].real[n][k] = cur_op[n, k].real
                        (<ComplexMatrix4*>self._ops)[m].imag[n][k] = cur_op[n, k].imag
        else:
            self._ops = malloc(self._num_ops * sizeof(ComplexMatrixN))
            for m in range(self._num_ops):
                cur_op = operators[m]
                (<ComplexMatrixN*>self._ops)[m] = createComplexMatrixN(self._num_targets)
                for k in range(mat_dim):
                    for n in range(mat_dim):
                        (<ComplexMatrixN*>self._ops)[m].real[n][k] = cur_op[n, k].real
                        (<ComplexMatrixN*>self._ops)[m].imag[n][k] = cur_op[n, k].imag

    @property
    def operators(self):
        cdef size_t mat_dim = 1 << self._num_targets
        cdef qcomp[:, :, :] np_ops = np.ndarray((self._num_ops, mat_dim, mat_dim), dtype=pyquest.core.np_qcomp)
        cdef size_t k, m, n
        for k in range(self._num_ops):
            if self._num_targets == 1:
                for m in range(mat_dim):
                    for n in range(mat_dim):
                        np_ops[k, m, n] = (
                            (<ComplexMatrix2*>self._ops)[k].real[m][n]
                            + 1j * (<ComplexMatrix2*>self._ops)[k].imag[m][n])
            elif self._num_targets == 2:
                for m in range(mat_dim):
                    for n in range(mat_dim):
                        np_ops[k, m, n] = (
                            (<ComplexMatrix4*>self._ops)[k].real[m][n]
                            + 1j * (<ComplexMatrix4*>self._ops)[k].imag[m][n])
            else:
                for m in range(mat_dim):
                    for n in range(mat_dim):
                        np_ops[k, m, n] = (
                            (<ComplexMatrixN*>self._ops)[k].real[m][n]
                            + 1j * (<ComplexMatrixN*>self._ops)[k].imag[m][n])
        return np_ops.base

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_targets == 1:
            quest.mixKrausMap(c_register, self._targets[0], <ComplexMatrix2*>self._ops, self._num_ops)
        elif self._num_targets == 2:
            quest.mixTwoQubitKrausMap(c_register, self._targets[0], self._targets[1], <ComplexMatrix4*>self._ops, self._num_ops)
        else:
            quest.mixMultiQubitKrausMap(c_register, self._targets, self._num_targets, <ComplexMatrixN*>self._ops, self._num_ops)

    def __dealloc__(self):
        cdef int m
        if self._num_targets > 2:
            for m in range(self._num_ops):
                destroyComplexMatrixN((<ComplexMatrixN*>self._ops)[m])
        free(self._ops)


cdef class PauliNoise(SingleQubitOperator):
    def __cinit__(self, target, probs):
        self.TYPE = OP_TYPES.OP_PAULI_NOISE
        self._prob_x = probs[0]
        self._prob_y = probs[1]
        self._prob_z = probs[2]

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.mixPauli(c_register, self._target,
                       self._prob_x, self._prob_y, self._prob_z)


cdef class MixDensityMatrix(GlobalOperator):
    def __cinit__(self, prob, Register density_matrix, copy_register=True):
        self.TYPE = OP_TYPES.OP_MIX_DENSITY
        self._prob = prob
        if not density_matrix.is_density_matrix:
            raise ValueError("Register 'density_matrix' must be "
                             "a density matrix.")
        if copy_register:
            self._other_register = density_matrix.copy()
        else:
            self._other_register = density_matrix

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.mixDensityMatrix(c_register, self._prob,
                               self._other_register.c_register)
