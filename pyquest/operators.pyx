"""Classes to represent generic operations in pyquest.

The most generic (possibly non-unitary, non-norm-preserving) operations
on quantum states are contained in this module.

Classes:
    BaseOperator:
        Abstract generic base class for all operators.
    ControlledOperator:
        Base class for all controlled operators.
    SingleQubitOperator:
        Base class for operators acting on a single qubit.
    MultiQubitOperator:
        Base class for operators potentially acting on multiple qubits.
    MatrixOperator:
        Most generic operator, specified by its matrix representation.
    PauliSum:
        A weighted sum of ``PauliProduct``s.
"""

import enum
import logging
from warnings import warn
import cython
import numpy as np
import pyquest
from pyquest.quest_error import QuESTError
from pyquest.unitaries import PauliProduct, PauliOperator

logger = logging.getLogger(__name__)


cdef class BaseOperator:
    """Generic base class for every class representing an operator.

    All operators on a quantum state must be derived from this class in
    order to work with pyquest. It only provides prototypes for the most
    basic tasks and returns ``NotImplementedError``s if not properly
    overridden.

    Cython attributes:
        TYPE (pyquest.quest_interface.OP_TYPES): Identifier of the
            operation for faster type checking than with
            ``IsInstance()``.
    """

    def __cinit__(self, *args, **kwargs):
        """Minimal set-up for a basic operator.

        Set TYPE to ``OP_ABSTRACT`` to identify instances of not
        properly implemented operator classes. All *args and *kwargs
        are ignored.
        """
        self.TYPE = OP_TYPES.OP_ABSTRACT

    def __repr__(self):
        return type(self).__name__ + "()"

    @property
    def inverse(self):
        """Calculate and return the inverse of an operator."""
        raise NotImplementedError(
            "inverse not implemented for this operator.")

    @property
    def targets(self):
        """Return a list of qubits this operator acts on."""
        raise NotImplementedError(
            "targets not implemented for this operator.")

    cpdef as_matrix(self, num_qubits=None):
        """Return the operator in matrix representation."""
        if num_qubits is None:
            try:
                num_qubits = max(self.targets) + 1  # 0-based index
            except NotImplementedError:
                raise ValueError(
                    "Number of qubits must be given for this operator.")
        cdef QuESTEnv *env_ptr = <QuESTEnv*>PyCapsule_GetPointer(
            pyquest.env.env_capsule, NULL)
        cdef Qureg tmp_reg = quest.createQureg(
            num_qubits, env_ptr[0])
        cdef long long k, m
        cdef long long mat_dim = 1LL << num_qubits
        cdef Complex amp
        cdef qcomp[:, :] res_mat = np.ndarray(
            (mat_dim, mat_dim), dtype=pyquest.core.np_qcomp)
        for k in range(mat_dim):
            quest.initClassicalState(tmp_reg, k)
            self.apply_to(tmp_reg)
            for m in range(mat_dim):
                amp = quest.getAmp(tmp_reg, m)
                res_mat[m, k].real = amp.real
                res_mat[m, k].imag = amp.imag
        quest.destroyQureg(tmp_reg, env_ptr[0])
        return res_mat.base

    cdef int apply_to(self, Qureg c_register) except -1:
        """Apply the operator to a ``quest_interface.Qureg``."""
        raise NotImplementedError(
            str(self) + " does not have an apply_to() method.")


cdef class ControlledOperator(BaseOperator):
    """Class implementing functionality for controlled operations.

    Abstract base class handling the keyword-only ``controls`` argument
    in the constructor and providing an interface for accessing it.
    """

    def __cinit__(self, *args, controls=None, **kwargs):
        self.controls = controls

    def __dealloc__(self):
        free(self._controls)

    def __repr__(self):
        if self._num_controls > 0:
            return type(self).__name__ + "(controls=" + str(self.controls) + ")"
        else:
            return type(self).__name__ + "()"

    @property
    def controls(self):
        cdef size_t k
        cdef list py_ctrls = [0] * self._num_controls
        for k in range(self._num_controls):
            py_ctrls[k] = self._controls[k]
        return py_ctrls

    @controls.setter
    def controls(self, value):
        cdef size_t k
        if value is None:
            self._num_controls = 0
            self._controls = NULL
            return
        try:
            self._num_controls = len(value)
            self._controls = <int*>malloc(self._num_controls
                                          * sizeof(self._controls[0]))
            for k in range(self._num_controls):
                self._controls[k] = value[k]
        except TypeError:
            try:
                # free() just in case the try branch failed after malloc
                free(self._controls)
                self._num_controls = 1
                self._controls = <int*>malloc(sizeof(self._controls[0]))
                self._controls[0] = value
            except TypeError:
                raise TypeError("Only integers and indexables of integers "
                                "are valid controls.")


cdef class SingleQubitOperator(ControlledOperator):
    """Class implementing an operator with only one target qubit.

    Abstract base class for all operator acting on a single qubit only.
    Provides properties for accessing the target qubit.
    """

    def __cinit__(self, target, *args, **kwargs):
        try:
            self._target = target
        except TypeError:
            self._target = target[0]
            if len(target) > 1:
                logger.warning(
                    "More than one target passed to single-qubit "
                    "operator. Using first target only.")

    def __repr__(self):
        res = type(self).__name__ + "(" + str(self.target)
        if self._num_controls > 0:
            res += ", controls=" + str(self.controls)
        return res + ")"

    def __reduce__(self):
        return (self.__class__, (self.target, ), self.controls)

    def __setstate__(self, state):
        self.controls = state

    @property
    def target(self):
        return self._target

    @property
    def targets(self):
        return [self._target]


cdef class MultiQubitOperator(ControlledOperator):
    """Class implementing an operator with multiple target qubits.

    Abstract base class for operators with (potentially) more than one
    target qubit. Provides functionality for accessing the target
    qubits.
    """

    def __cinit__(self, targets=None, *args, target=None, **kwargs):
        if targets is not None and target is not None:
            raise ValueError("Only one of the keyword arguments 'targets' "
                             "and 'target' may be supplied.")
        if targets is None:
            targets = target
        if targets is None:
            raise ValueError("One of the keyword arguments 'targets' "
                             "or 'target' must be supplied.")
        # Single- and Multi-qubit operators have their try-except
        # cases reversed such that the fastest operation is the
        # encouraged one.
        try:
            self._num_targets = len(targets)
            self._targets = <int*>malloc(self._num_targets
                                         * sizeof(self._targets[0]))
            for k in range(self._num_targets):
                self._targets[k] = targets[k]
        # If we got a scalar, there is only 1 qubit on which to act.
        except TypeError:
            try:
                self._num_targets = 1
                free(self._targets)  # In case we malloced in try.
                self._targets = <int*>malloc(sizeof(self._targets[0]))
                self._targets[0] = targets
            except TypeError:
                raise TypeError("Only integers and indexables of integers "
                                "are valid targets.")

    def __dealloc__(self):
        free(self._targets)

    def __reduce__(self):
        return (self.__class__, (self.targets, ), self.controls)

    def __setstate__(self, state):
        self.controls = state

    def __repr__(self):
        res = type(self).__name__ + "(" + str(self.targets)
        if self._num_controls > 0:
            res += ", controls=" + str(self.controls)
        return res + ")"

    @property
    def targets(self):
        cdef size_t k
        py_targs = [0] * self._num_targets
        for k in range(self._num_targets):
            py_targs[k] = self._targets[k]
        return py_targs

    @property
    def target(self):
        if self._num_targets == 1:
            return self._targets[0]
        return self.targets


cdef class MatrixOperator(MultiQubitOperator):

    def __cinit__(self, targets=None, matrix=None, controls=None, target=None,
                  **kwargs):
        self.TYPE = OP_TYPES.OP_MATRIX
        if matrix is None:
            raise TypeError("Matrix representation must be given.")
        cdef size_t matrix_dim = 2 ** self._num_targets
        if isinstance(matrix, np.ndarray):
            if matrix.ndim != 2:
                raise ValueError("Array must have exactly 2 dimensions.")
            if matrix.shape[0] != matrix_dim or matrix.shape[1] != matrix_dim:
                raise ValueError("Matrix dimension must be "
                                 "2 ** num_qubits x 2 ** num_qubits.")
        else:
            raise NotImplementedError("Only numpy.ndarray matrices are "
                                      "currently supported.")

    def __init__(self, targets=None, matrix=None, controls=None, target=None,
                 **kwargs):
        # Even though these are C-level initialisers, call them in
        # __init__() rather than __cinit__() so they can be overridden
        # by subclasses.
        self._create_array_property()
        self._numpy_array_to_matrix_attribute(matrix)

    def __dealloc__(self):
        # For 1 or 2 qubits the matrix elements are directly in the
        # _matrix attribute, so they are freed togehter with _matrix.
        if self._num_targets > 2 or self._num_controls > 0:
            if self._matrix != NULL:
                destroyComplexMatrixN((<ComplexMatrixN*>self._matrix)[0])
        else:
            # The arrays of the row-pointers for 1 and 2 qubits
            # are allocated manually, because the arrays themselves
            # are in the _matrix variable.
            free(self._real)
            free(self._imag)
        free(self._matrix)

    def __repr__(self):
        res = type(self).__name__ + "(\n    " + str(self.targets) + ","

        cdef size_t matrix_dim = 2 ** self._num_targets
        cdef size_t i, j
        res += "\n    array(\n        ["
        for i in range(matrix_dim):
            res += "["
            for j in range(matrix_dim):
                res += f'{self._real[i][j]:f}{self._imag[i][j]:+f}j, '
            res = res[:-2] + "],\n         "
        res = res[:-11] + "])"
        if self._num_controls > 0:
            res += ",\n    controls=" + str(self.controls)
        return res + ")"

    cdef int apply_to(self, Qureg c_register) except -1:
        if self._num_controls == 0:
            if self._num_targets == 1:
                quest.applyMatrix2(
                    c_register, self._targets[0],
                    (<ComplexMatrix2*>self._matrix)[0])
            elif self._num_targets == 2:
                quest.applyMatrix4(
                    c_register, self._targets[0], self._targets[1],
                    (<ComplexMatrix4*>self._matrix)[0])
            else:
                quest.applyMatrixN(
                    c_register, self._targets, self._num_targets,
                    (<ComplexMatrixN*>self._matrix)[0])
        else:
            quest.applyMultiControlledMatrixN(
                c_register, self._controls, self._num_controls, self._targets,
                self._num_targets, (<ComplexMatrixN*>self._matrix)[0])

    cdef _create_array_property(self):
        cdef size_t matrix_dim = 2 ** self._num_targets
        cdef size_t k
        # The only cases where core QuEST supports ComplexMatrix2
        # or ComplexMatrix4 for generic matrices are non-controlled
        # cases.
        if self._num_targets == 1 and self._num_controls == 0:
            self._matrix = malloc(sizeof(ComplexMatrix2))
            self._real = <qreal**>malloc(matrix_dim * sizeof(self._real[0]))
            self._imag = <qreal**>malloc(matrix_dim * sizeof(self._imag[0]))
            for k in range(matrix_dim):
                self._real[k] = (<ComplexMatrix2*>self._matrix).real[k]
                self._imag[k] = (<ComplexMatrix2*>self._matrix).imag[k]
        elif self._num_targets == 2 and self._num_controls == 0:
            self._matrix = malloc(sizeof(ComplexMatrix4))
            self._real = <qreal**>malloc(matrix_dim * sizeof(self._real[0]))
            self._imag = <qreal**>malloc(matrix_dim * sizeof(self._imag[0]))
            for k in range(matrix_dim):
                self._real[k] = (<ComplexMatrix4*>self._matrix).real[k]
                self._imag[k] = (<ComplexMatrix4*>self._matrix).imag[k]
        else:
            self._matrix = malloc(sizeof(ComplexMatrixN))
            (<ComplexMatrixN*>self._matrix)[0] = createComplexMatrixN(self._num_targets)
            self._real = (<ComplexMatrixN*>self._matrix).real
            self._imag = (<ComplexMatrixN*>self._matrix).imag

    cdef _numpy_array_to_matrix_attribute(self, np.ndarray arr):
        # For typed memoryviews we need to call different methods
        # depending on the dtype of the array.
        if arr.dtype == np.single:
            self._copy_single_array(arr)
        elif arr.dtype == np.double:
            self._copy_double_array(arr)
        elif arr.dtype == np.longdouble:
            self._copy_longdouble_array(arr)
        elif arr.dtype == np.csingle:
            self._copy_csingle_array(arr)
        elif arr.dtype == np.cdouble:
            self._copy_cdouble_array(arr)
        elif arr.dtype == np.clongdouble:
            self._copy_clongdouble_array(arr)
        else:
            # For unsupported types we use the (slow) generic copy
            # through the Python API as a fallback.
            self._copy_generic_array(arr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_generic_array(self, np.ndarray arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m].real
                self._imag[k][m] = arr[k, m].imag

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_single_array(self, float[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m]
                self._imag[k][m] = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_double_array(self, double[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m]
                self._imag[k][m] = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_longdouble_array(self, long double[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m]
                self._imag[k][m] = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_csingle_array(self, float complex[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m].real
                self._imag[k][m] = arr[k, m].imag

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_cdouble_array(self, double complex[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m].real
                self._imag[k][m] = arr[k, m].imag

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _copy_clongdouble_array(self, long double complex[:, :] arr):
        cdef size_t k, m
        for k in range(arr.shape[0]):
            for m in range(arr.shape[1]):
                self._real[k][m] = arr[k, m].real
                self._imag[k][m] = arr[k, m].imag


cdef class PauliSum(GlobalOperator):

    _PAULI_REPR = {0: "", 1: "X", 2: "Y", 3: "Z"}
    _MULTIPLICATION_TABLE = np.array([[0, 1, 2, 3],
                                      [1, 0, 3, 2],
                                      [2, 3, 0, 1],
                                      [3, 2, 1, 0]], dtype=np.intc)

    _MULTIPLICATION_FACTORS = np.array([[0, 0, 0, 0],
                                        [0, 0, 1, 3],
                                        [0, 3, 0, 1],
                                        [0, 1, 3, 0]], dtype=np.intc)

    def __cinit__(self, pauli_terms=None):
        self.TYPE = OP_TYPES.OP_PAULI_SUM
        self._root = <PauliNode*>calloc(1, sizeof(PauliNode))
        if pauli_terms is None:
            return
        # Offload all the conversion from various types to the
        # constructor of PauliProduct.
        try:
            # First try to create a PauliProduct from whatever we got.
            if not isinstance(pauli_terms, PauliProduct):
                prod = PauliProduct(pauli_terms)
            # If pauli_terms can be cast to a PauliProduct, its
            # coefficient is implicitly assumed to be 1.
            pauli_terms = ((1, prod),)
        except (ValueError, TypeError):
            pass
        try:
            for cur_term in pauli_terms:
                self._add_term_from_PauliProduct(
                    cur_term[0],
                    cur_term[1] if isinstance(cur_term[1], PauliProduct)
                                else PauliProduct(cur_term[1]))
        except TypeError as exc:
            raise TypeError("Unsupported type encountered in pauli_terms") from exc

    def __str__(self):
        return PauliSum._get_node_string(self._root, "", 0)[3:]

    def __repr__(self):
        return f"{type(self).__name__}({self.terms})"

    def __mul__(PauliSum self, PauliSum other):
        res = PauliSum()
        PauliSum._multiply_subtrees(self._root, other._root, res._root, 0)
        return res

    def __add__(PauliSum self, PauliSum other):
        res = PauliSum()
        res._root.coefficient = self._root.coefficient + other._root.coefficient
        PauliSum._add_subtree(self._root, res._root, 1)
        PauliSum._add_subtree(other._root, res._root, 1)
        return res

    def __iadd__(PauliSum self, PauliSum other):
        self._root.coefficient += other._root.coefficient
        PauliSum._add_subtree(other._root, self._root, 1)
        return self

    def __dealloc__(self):
        PauliSum._free_subtree(self._root)
        free(self._root)

    @property
    def num_terms(self):
        return PauliSum._num_subtree_terms(self._root)

    @property
    def num_qubits(self):
        # Subtract 1 because the root alone doesn't need any qubits
        return PauliSum._num_subtree_qubits(self._root) - 1

    @property
    def terms(self):
        return self._node_terms(self._root, [])

    cpdef void round(PauliSum self, qreal eps):
        PauliSum._round_node(self._root, eps)
        self._cache_valid = 0

    cpdef void compress(PauliSum self, qreal eps):
        PauliSum._compress_node(self._root, eps)
        self._cache_valid = 0

    cdef int apply_to(self, Qureg c_register) except -1:
        cdef int num_terms = PauliSum._num_subtree_terms(self._root)
        cdef int num_qubits = PauliSum._num_subtree_qubits(self._root) - 1
        if c_register.numQubitsRepresented < num_qubits:
            raise ValueError(f"Register does not have enough qubits. Needs {num_qubits}, "
                             f"but only has {c_register.numQubitsRepresented}.")
        # QuEST needs the number of qubits in the Qureg
        num_qubits = c_register.numQubitsRepresented
        cdef pauliOpType *prefix = <pauliOpType*>malloc(num_qubits * sizeof(prefix[0]))
        # Only ever increase allocated memory, there is no point in
        # shrinking it just because the PauliSum is applied to a
        # smaller Register.
        if num_terms > self._quest_hamil.numSumTerms:
            free(self._quest_hamil.termCoeffs)
            self._quest_hamil.termCoeffs = <qreal*>malloc(num_terms * sizeof(self._quest_hamil.termCoeffs[0]))
            self._cache_valid = 0
            self._num_coeffs_allocated = num_terms
        if (num_terms * num_qubits > self._num_paulis_allocated):
            free(self._quest_hamil.pauliCodes)
            self._quest_hamil.pauliCodes = <pauliOpType*>malloc(num_terms * num_qubits * sizeof(self._quest_hamil.pauliCodes[0]))
            self._cache_valid = 0
            self._num_paulis_allocated = num_terms * num_qubits
        self._quest_hamil.numQubits = num_qubits
        self._quest_hamil.numSumTerms = num_terms
        cdef pauliOpType *pauli_ptr = self._quest_hamil.pauliCodes
        cdef qreal *coeff_ptr = self._quest_hamil.termCoeffs
        if not self._cache_valid:
            PauliSum._expand_subtree_terms(self._root, num_qubits, prefix, 0, &coeff_ptr, &pauli_ptr)
            self._cache_valid = 1
        cdef QuESTEnv *env_ptr = <QuESTEnv*>PyCapsule_GetPointer(pyquest.env.env_capsule, NULL)
        cdef Qureg temp_reg = quest.createCloneQureg(c_register, env_ptr[0])
        quest.applyPauliHamil(temp_reg, self._quest_hamil, c_register)

    cdef int _add_term_from_PauliProduct(self, coeff, pauli_product) except -1:
        cdef int paulis[MAX_QUBITS]
        cdef int n = 0
        # FIXME: This is probably inefficient, because it passes the
        #        pauli_types through a Python list. cimporting
        #        PauliProduct would fix this, but will create circular
        #        imports. Redesign needed for speedup.
        for cur_pauli in pauli_product.pauli_types:
            if n > MAX_QUBITS:
                raise ValueError("Trying to add Pauli terms on too many qubits")
            paulis[n] = cur_pauli
            n += 1
        PauliSum._add_term(self._root, coeff, paulis, n)

    cdef list _node_terms(PauliSum self, PauliNode* node, list prefix_list):
        term_list = []
        if node == NULL:
            return term_list
        if node[0].coefficient != 0:
            term_list.append((node[0].coefficient, prefix_list))
        for k in range(4):
            if node[0].children[k] != NULL:
                term_list.extend(self._node_terms(node[0].children[k], prefix_list + [k]))
        return term_list

    @staticmethod
    cdef int _num_subtree_terms(PauliNode *node):
        if node == NULL:
            return 0
        cdef int num = node.coefficient != 0
        cdef int k
        for k in range(4):
            if node.children[k] == NULL:
                continue
            num += PauliSum._num_subtree_terms(node.children[k])
        return num

    @staticmethod
    cdef int _num_subtree_qubits(PauliNode *node):
        if node == NULL:
            return 0
        cdef int qubits_after = 0
        cdef int k
        for k in range(4):
            if node.children[k] == NULL:
                continue
            qubits_after = max(qubits_after, PauliSum._num_subtree_qubits(node.children[k]))
        return 1 + qubits_after

    @staticmethod
    cdef void _expand_subtree_terms(
        PauliNode *node, int num_qubits, pauliOpType* prefix, int num_prefix,
        qreal **coefficients, pauliOpType **pauli_terms):
        cdef int k, n
        if node.coefficient != 0:
            for k in range(num_prefix):
                pauli_terms[0][k] = prefix[k]
            for k in range(num_prefix, num_qubits):
                pauli_terms[0][k] = pauliOpType.PAULI_I
            pauli_terms[0] = &pauli_terms[0][num_qubits]
            coefficients[0][0] = node.coefficient.real
            coefficients[0] = &coefficients[0][1]
        for n in range(4):
            if node.children[n] == NULL:
                continue
            # This is probably safe on most compilers, but there might
            # be a more elegant way to do this.
            prefix[num_prefix] = <pauliOpType>n
            PauliSum._expand_subtree_terms(
                node.children[n], num_qubits, prefix, num_prefix + 1,
                coefficients, pauli_terms)

    @staticmethod
    cdef void _add_term(PauliNode *root, qcomp coeff, int *paulis, int num_qubits):
        self._cache_valid = 0
        cdef int k
        cdef PauliNode *cur_node = root
        for k in range(num_qubits):
            if cur_node[0].children[paulis[k]] == NULL:
                cur_node[0].children[paulis[k]] = <PauliNode*>calloc(1, sizeof(PauliNode))
            cur_node = cur_node.children[paulis[k]]
        cur_node[0].coefficient += coeff

    @staticmethod
    cdef str _get_node_string(PauliNode* node, str pauli_prefix, int cur_qubit):
        if node == NULL:
            return ""
        if node.coefficient == 0:
            res_str = ""
        # This whole section is only necessary because Python can't
        # compactly print complex numbers on its own.
        elif (node.coefficient.real == 0):
            if node.coefficient.imag < 0 or node.coefficient.real < 0:
                res_str = f" - {-node.coefficient.imag}j{pauli_prefix}"
            else:
                res_str = f" + {node.coefficient.imag}j{pauli_prefix}"
        elif (node.coefficient.imag == 0):
            if node.coefficient.real < 0:
                res_str = f" - {-node.coefficient.real}{pauli_prefix}"
            else:
                res_str = f" + {node.coefficient.real}{pauli_prefix}"
        else:
            res_str =  f" + {node[0].coefficient}{pauli_prefix}"
        res_str += PauliSum._get_node_string(node[0].children[0], pauli_prefix, cur_qubit + 1)
        for k in range(1, 4):
            res_str += PauliSum._get_node_string(node[0].children[k], f"{pauli_prefix} {PauliSum._PAULI_REPR[k]}{cur_qubit}", cur_qubit + 1)
        return res_str

    @staticmethod
    cdef void _multiply_subtrees(PauliNode *left, PauliNode *right, PauliNode *product, int prefact):
        self._cache_valid = 0
        product.coefficient += left.coefficient * right.coefficient * 1j ** prefact
        if left.coefficient != 0:
            PauliSum._add_subtree(right, product, left.coefficient * 1j ** prefact)
        if right.coefficient != 0:
            PauliSum._add_subtree(left, product, right.coefficient * 1j ** prefact)
        cdef int k, m, add_prefactor
        for k in range(4):
            if left.children[k] != NULL:
                for m in range(4):
                    if right.children[m] != NULL:
                        # Potentially slow Python member retrieval
                        cur_prod_pauli = PauliSum._MULTIPLICATION_TABLE[k, m]
                        add_prefactor = PauliSum._MULTIPLICATION_FACTORS[k, m]
                        if product.children[cur_prod_pauli] == NULL:
                            product.children[cur_prod_pauli] = <PauliNode*>calloc(1, sizeof(PauliNode))
                        PauliSum._multiply_subtrees(
                                left.children[k], right.children[m],
                                product.children[cur_prod_pauli], (prefact + add_prefactor) % 4)

    @staticmethod
    cdef void _add_subtree(PauliNode *source, PauliNode *target, qcomp factor):
        self._cache_valid = 0
        cdef int k
        for k in range(4):
            if source.children[k] == NULL:
                continue
            if target.children[k] == NULL:
                target.children[k] = <PauliNode*>calloc(1, sizeof(PauliNode))
            target.children[k].coefficient += source.children[k].coefficient * factor
            PauliSum._add_subtree(source.children[k], target.children[k], factor)

    @staticmethod
    cdef void _round_node(PauliNode *node, qreal eps):
        self._cache_valid = 0
        if abs(node.coefficient.real) < eps:
            node.coefficient.real = 0
        if abs(node.coefficient.imag) < eps:
            node.coefficient.imag = 0
        cdef int k
        for k in range(4):
            if node.children[k] == NULL:
                continue
            PauliSum._round_node(node.children[k], eps)

    @staticmethod
    cdef int _compress_node(PauliNode *node, qreal eps):
        # Returns 1 if the node has no more children after compressing,
        # 0 otherwise.
        self._cache_valid = 0
        cdef int no_children = 1
        cdef int k
        if node.children[0] != NULL:
            # Identity terms should propagate upwards as far as possible.
            no_children = PauliSum._compress_node(node.children[0], eps)
            node.coefficient += node.children[0].coefficient
            node.children[0].coefficient = 0
            if no_children:
                free(node.children[0])
                node.children[0] = NULL
        for k in range(1, 4):
            if node.children[k] == NULL:
                continue
            if PauliSum._compress_node(node.children[k], eps):
                if abs(node.children[k].coefficient) < eps:
                    free(node.children[k])
                    node.children[k] = NULL
                else:
                    no_children = 0
        return no_children

    @staticmethod
    cdef void _free_subtree(PauliNode *node):
        if node == NULL:
            return
        cdef int k
        for k in range(4):
            if node.children[k] == NULL:
                continue
            PauliSum._free_subtree(node.children[k])
            free(node.children[k])
            node.children[k] = NULL


cdef class TrotterCircuit(GlobalOperator):

    def __cinit__(self):
        self.TYPE = OP_TYPES.OP_TROTTER_CIRC
        raise NotImplementedError("TrotterCircuit not yet implemented.")


cdef class DiagonalOperator(GlobalOperator):

    def __cinit__(self, num_qubits, diag_elements=None):
        cdef QuESTEnv *env_ptr = <QuESTEnv*>PyCapsule_GetPointer(
            pyquest.env.env_capsule, NULL)
        cdef qreal[:] real_el, imag_el
        self.TYPE = OP_TYPES.OP_DIAGONAL
        self._diag_op = quest.createDiagonalOp(num_qubits, env_ptr[0])
        if diag_elements is not None:
            # We'll try to handle anything that can be an ndarray by
            # just giving it to the np.array constructor.
            if not isinstance(diag_elements, np.ndarray):
                diag_elements = np.array(diag_elements)
            diag_elements = diag_elements.squeeze()
            if diag_elements.ndim != 1:
                raise ValueError("'diag_elements' must only have one "
                                 "dimension.")
            if diag_elements.size != 1 << self._diag_op.numQubits:
                raise ValueError("'diag_elements' must have length of "
                                 "2**num_qubits")
            # FIXME This possibly creates copies of real and imaginary
            #       parts (QuEST requires contiguous arrays for real and
            #       imag separately), so doing it in chunks or write
            #       straight to memory would be better.
            real_el = np.require(
                diag_elements.real, dtype=pyquest.core.np_qreal,
                requirements='CW')
            imag_el = np.require(
                diag_elements.imag, dtype=pyquest.core.np_qreal,
                requirements='CW')
            quest.initDiagonalOp(self._diag_op, &real_el[0], &imag_el[0])

    def __dealloc__(self):
        cdef QuESTEnv *env_ptr = <QuESTEnv*>PyCapsule_GetPointer(
            pyquest.env.env_capsule, NULL)
        quest.destroyDiagonalOp(self._diag_op, env_ptr[0])

    def __repr__(self):
        return ("<" + type(self).__name__ + " on "
                + str(self._diag_op.numQubits) + " qubits at "
                + hex(id(self)) + ">")

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.applyDiagonalOp(c_register, self._diag_op)


class _BitEncoding(enum.IntEnum):
    UNSIGNED = quest.bitEncoding.UNSIGNED
    TWOS_COMPLEMENT = quest.bitEncoding.TWOS_COMPLEMENT


class _PhaseFuncType(enum.IntEnum):
    # Careful when adding new functions to this enum, the PhaseFunc
    # constructor, __repr__, and inverse rely on the "SCALED" and
    # "INVERSE" naming conventions to determine the correct structure
    # of parameters to pass to QuEST.
    NORM = quest.phaseFunc.NORM
    SCALED_NORM = quest.phaseFunc.SCALED_NORM
    INVERSE_NORM = quest.phaseFunc.INVERSE_NORM
    SCALED_INVERSE_NORM = quest.phaseFunc.SCALED_INVERSE_NORM
    SCALED_INVERSE_SHIFTED_NORM = quest.phaseFunc.SCALED_INVERSE_SHIFTED_NORM
    PRODUCT = quest.phaseFunc.PRODUCT
    SCALED_PRODUCT = quest.phaseFunc.SCALED_PRODUCT
    INVERSE_PRODUCT = quest.phaseFunc.INVERSE_PRODUCT
    SCALED_INVERSE_PRODUCT = quest.phaseFunc.SCALED_INVERSE_PRODUCT
    DISTANCE = quest.phaseFunc.DISTANCE
    SCALED_DISTANCE = quest.phaseFunc.SCALED_DISTANCE
    INVERSE_DISTANCE = quest.phaseFunc.INVERSE_DISTANCE
    SCALED_INVERSE_DISTANCE = quest.phaseFunc.SCALED_INVERSE_DISTANCE
    SCALED_INVERSE_SHIFTED_DISTANCE = quest.phaseFunc.SCALED_INVERSE_SHIFTED_DISTANCE
    EXPONENTIAL_POLYNOMIAL = enum.auto()


# Extension types do not allow nested classes, so they are defined as
# "private" in the module scope and referenced in the PhaseFunc
# extension type. The __str__ and __repr__ methods of IntEnum use the
# __name__ attribute for printing, so remove the underscore here for
# nicer and less confusing prints.
_BitEncoding.__name__ = "BitEncoding"
_PhaseFuncType.__name__ = "FuncType"


cdef class PhaseFunc(GlobalOperator):

    BitEncoding = _BitEncoding
    FuncType = _PhaseFuncType

    # Maximum number of parameters for any named function, excluding
    # sub-register shifts.
    _MAX_NUM_PARAMETERS = 2

    def __cinit__(
            self, targets, func_type='exponential_polynomial',
            bit_encoding='unsigned', overrides=None, **kwargs):
        """Create a new phase function with specific parameters.

        Args:
            targets (iterable[int] | iterable[iterable[int]]): Qubits
                the phase function should be applied to. If given as
                iterable of `int`s, specifies the qubits which,
                interpreted as a binary number, serve as the input to
                a single-variable phase function. The first qubit in the
                iterable is the least significant bit in its binary
                interpretation. A state with qubit 0 and qubit 1 in
                state 0, and qubit 2 in state 1 is interpreted as a
                decimal 4 if targets=(0, 1, 2), but as a decimal 1 if
                targets=(2, 1, 0) (when using unsigned binary
                representation). The qubit indices do not need to be
                in strictly increasing or decreasing order.

                For multi-variable functions, `targets` must be a
                nested iterable, where each element is an iterable of
                integers. Each of the elements in `targets` then
                represents a sub-register in the same fashion as
                described above, and each such sub-register serves as
                one variable in a multi-variable function. How exactly
                they are used depends on the function type.
            func_type (PhaseFunc.FuncType | str): The type of phase
                function to apply. Must be either directly from enum
                PhaseFunc.FuncType or a string matching one of the
                elements therein. Defaults to 'exponential_polynomial'.
            bit_encoding (PhaseFunc.BitEncoding | str): The encoding
                by which the binary representation in the registers
                is interpreted. Must be either directly from enum
                PhaseFunc.BitEncoding or a string matching one of the
                elements therein. Defaults to 'unsigned'.
            overrides (dict[int: float] | dict[tuple[int]: float]):
                Manually replaces the function output for certain
                input values. For single-valued functions, the keys
                can be integers and directly specify for which inputs
                the function output should be replaced. The value of
                each dictionary element gives the phase to be applied
                to the matching input. Passing `overrides={4: 3.14}`
                would multiply the coefficient of the basis state
                corresponding to decimal 4 (in the chosen binary
                encoding) with exp(3.14j).

                For multi-valued functions, the keys of the dictionary
                must be tuples, where entry `k` of the tuple
                corresponds to the state the `k`th sub-register must
                be in to be overridden. When working with two
                sub-registers, `overrides={(3, 5): 3.14}` would effect
                a phase of exp(3.14j) on a state where the first
                sub-register is in a state corresponding to decimal 3,
                and the state of the second sub-register corresponds to
                decimal 5 simultaneously. The order of the sub-registers
                is defined by the order they appear in the `targets`
                argument. Defaults to None.

        Keyword Args:
            terms (iterable[iterable[int]]): The function type
                'exponential_polynomial' requires the polynomial to be
                passed to this keyword argument. The exponential
                polynomial is a sum of terms, and, each term is of the
                form `c * (x_r)**k`. This is represented by an iterable
                of length 3 of the form (c, r, k). The first element is
                the coefficient, the second element is the index of the
                sub-register to be used, and the third element is the
                exponent. `terms` is an iterable, where each element is
                such a term.

                The exponential polynomial
                    f(r) = r0**2 + 2 * r1 + 4 * r1**(-1) - 3.1 * r2**0.5
                acting on three sub-registers would be specified by
                passing
                    `terms=[(1, 0, 2), (2, 1, 1),
                            (4, 1, -1), (-3.1, 2, .5)]`
                to the constructor.
            scaling (float): Function types containing `SCALED` in their
                name require this costant factor by which the function
                values are multiplied. Not allowed for any other
                function types.
            divergence_override (float): Allows automatic overriding of
                function values at divergent points. Only available for
                function types containing `INVERSE` in their name, in
                which it replaces occurances where the denominator is
                zero with the specified value `divergence_override`,
                effectively multiplying their coefficients with
                exp(1j * divergence_override). Defaults to 0.
            shifts (iterable[float]): Function types containing
                `SHIFTED` require this argument to contain the amounts
                by which the sub-registers (or pairs thereof) will be
                shifted. For `SCALED_INVERSE_SHIFTED_NORM` there must be
                as many elements in this argument as there are
                sub-registers, `SCALED_INVERSE_SHIFTED_DISTANCE`
                requires exactly half as many shifts as there are
                sub-registers.
        """
        cdef int k, m, n
        cdef int num_qubits_in_regs, qubit_counter
        cdef int num_terms, term_counter

        if not isinstance(func_type, PhaseFunc.FuncType):
            try:
                func_type = PhaseFunc.FuncType[func_type.upper()]
            except KeyError:
                raise ValueError(f'"{func_type}" is not a valid function type.') from None
        self._phase_func_type = func_type
        if func_type is PhaseFunc.FuncType.EXPONENTIAL_POLYNOMIAL:
            self._is_poly = 1  # For quicker check in apply_to().

        if not isinstance(bit_encoding, PhaseFunc.BitEncoding):
            try:
                bit_encoding = PhaseFunc.BitEncoding[bit_encoding.upper()]
            except KeyError:
                raise ValueError(f'"{bit_encoding}" is not a valid bit encoding.') from None
        self._bit_encoding = bit_encoding

        # Pack single register spec into a tuple for consistent parsing.
        if isinstance(targets[0], int):
            targets = (targets,)
        self._num_regs = len(targets)
        num_qubits_in_regs = sum(len(sub_reg) for sub_reg in targets)
        self._num_qubits_per_reg = <int*>malloc(
            self._num_regs * sizeof(self._num_qubits_per_reg[0]))
        self._qubits_in_regs = <int*>malloc(
            num_qubits_in_regs * sizeof(self._qubits_in_regs[0]))
        qubit_counter = 0
        for k in range(self._num_regs):
            self._num_qubits_per_reg[k] = len(targets[k])
            for m in range(self._num_qubits_per_reg[k]):
                self._qubits_in_regs[qubit_counter] = targets[k][m]
                qubit_counter += 1

        if not overrides:
            self._num_overrides = 0
        else:
            self._num_overrides = len(overrides)
            self._override_inds = <long long int*>malloc(
                self._num_regs * self._num_overrides
                * sizeof(self._override_inds[0]))
            self._override_phases = <qreal*>malloc(
                self._num_overrides * sizeof(self._override_phases[0]))
            # The extraction of the override indices can lead to an
            # illegal object state if the keys in the 'overrides' dict
            # are of different lengths, so explicitly check that first.
            override_lens = set(1 if isinstance(override, int)
                                else len(override)
                                for override in overrides)
            if (len(override_lens) != 1
                    or override_lens.pop() != self._num_regs):
                raise ValueError("All keys of the 'override' dict must "
                                 "have a length equal to the number of "
                                 "sub-registers.")
            k = 0
            m = 0
            for cur_override_ind in overrides:
                if isinstance(cur_override_ind, int):
                    self._override_inds[m] = cur_override_ind
                    m += 1
                else:
                    for cur_override_sub_ind in cur_override_ind:
                        self._override_inds[m] = cur_override_sub_ind
                        m += 1
                self._override_phases[k] = overrides[cur_override_ind]
                k += 1

        if func_type is PhaseFunc.FuncType.EXPONENTIAL_POLYNOMIAL:
            if 'terms' not in kwargs:
                raise TypeError("Exponential polynomial needs "
                                "argument 'terms'.")
            terms = kwargs['terms']
            if not all([0 <= term[1] < self._num_regs
                       for term in terms]):
                raise ValueError("All register indices must be >= 0 "
                                 "and < number of sub-registers.")
            self._num_terms_per_reg = <int*>malloc(
                self._num_regs * sizeof(self._num_terms_per_reg[0]))
            num_terms = len(terms)
            self._exponents=<qreal*>malloc(sizeof(self._exponents[0]))
            self._exponents = <qreal*>malloc(
                num_terms * sizeof(self._exponents[0]))
            self._coeffs = <qreal*>malloc(
                num_terms * sizeof(self._coeffs[0]))
            term_counter = 0
            for k in range(self._num_regs):
                subreg_terms = [term for term in terms if term[1] == k]
                self._num_terms_per_reg[k] = len(subreg_terms)
                for term in subreg_terms:
                    self._coeffs[term_counter] = term[0]
                    self._exponents[term_counter] = term[2]
                    term_counter += 1
        elif 'terms' in kwargs and kwargs['terms'] is not None:
            raise TypeError("'terms' argument is only allowed for "
                            "exponential polynomials.")

        # Since we control the _PhaseFuncType enum and must port
        # additional functions manually, we can rely on functions
        # that support scaling to contain 'SCALED', and potentially
        # diverging functions to contain 'INVERSE'.
        if ('scaling' in kwargs
            and 'SCALED' not in func_type.name
            and kwargs['scaling'] is not None):
                raise TypeError("'scaling' argument is only allowed for "
                                "scalable functions.")

        if 'SCALED' in func_type.name and 'scaling' not in kwargs:
            raise TypeError("Scaled function needs 'scaling' "
                            "keyword argument.")

        if ('shifts' in kwargs
            and 'SHIFTED' not in func_type.name
            and kwargs['shifts'] is not None):
                raise TypeError("'shifts' argument is only allowed for "
                                "shifted functions.")

        if 'SHIFTED' in func_type.name and 'shifts' not in kwargs:
            raise TypeError("Shifted function needs 'shifts' keyword "
                            "argument")

        if ('divergence_override' in kwargs
            and 'INVERSE' not in func_type.name
            and kwargs['divergence_override'] is not None):
                raise TypeError(
                    "'divergence_override' only available for predefined "
                    "'INVERSE' functions. Specify manually via 'overrides' "
                    "for exponential polynomials.")

        if 'INVERSE' in func_type.name and 'divergence_override' not in kwargs:
            kwargs['divergence_override'] = 0

        # Just allocate as much memory as we can possibly need,
        # this is still negligible memory waste.
        self._parameters = <qreal*>malloc(
            (PhaseFunc._MAX_NUM_PARAMETERS + self._num_regs)
            * sizeof(self._parameters[0]))
        self._num_parameters = 0

        if 'SCALED' in func_type.name:
            self._parameters[self._num_parameters] = kwargs['scaling']
            self._num_parameters += 1

        if 'INVERSE' in func_type.name:
            self._parameters[self._num_parameters] = kwargs['divergence_override']
            self._num_parameters += 1

        if 'SHIFTED' in func_type.name:
            for shift in kwargs['shifts']:
                self._parameters[self._num_parameters] = shift
                self._num_parameters += 1

    def __dealloc__(self):
        free(self._override_inds)
        free(self._override_phases)
        free(self._parameters)
        free(self._num_qubits_per_reg)
        free(self._qubits_in_regs)
        free(self._num_terms_per_reg)
        free(self._exponents)
        free(self._coeffs)

    def __repr__(self):
        res = type(self).__name__ + "("
        res += f"targets={str(self.targets)}"
        res += f', func_type="{PhaseFunc.FuncType(self._phase_func_type).name.lower()}"'
        res += f', bit_encoding="{PhaseFunc.BitEncoding(self._bit_encoding).name.lower()}"'
        if self._num_overrides:
            res += ", overrides=" + str(self.overrides)
        if self._num_terms_per_reg:
            res += f", terms={self.terms}"
        if "SCALED" in PhaseFunc.FuncType(self._phase_func_type).name:
            res += f", scaling={self.scaling}"
        if "INVERSE" in PhaseFunc.FuncType(self._phase_func_type).name:
            res += f", divergence_override={self.divergence_override}"
        if "SHIFTED" in PhaseFunc.FuncType(self._phase_func_type).name:
            res += f", shifts={list(self.shifts)}"
        return res + ")"

    @property
    def overrides(self):
        if self._num_regs > 1:
            ind_iterator = iter(self._override_inds[k]
                                for k in range(self._num_regs * self._num_overrides))
            return {tuple(next(ind_iterator)
                          for _ in range(self._num_regs)):
                    self._override_phases[k]
                    for k in range(self._num_overrides)}
        else:
            return {self._override_inds[k]: self._override_phases[k]
                    for k in range(self._num_overrides)}

    @property
    def targets(self):
        num_reg_qubits = sum(self._num_qubits_per_reg[k]
                             for k in range(self._num_regs))
        reg_qubits = iter(self._qubits_in_regs[k]
                          for k in range(num_reg_qubits))
        if self._num_regs == 1:
            return tuple(reg_qubits)
        return tuple(tuple(next(reg_qubits) for m in range(self._num_qubits_per_reg[k]))
                     for k in range(self._num_regs))

    @property
    def terms(self):
        if self._num_terms_per_reg:
            num_terms = sum(self._num_terms_per_reg[k] for k in range(self._num_regs))
            coeffs = iter(self._coeffs[k] for k in range(num_terms))
            exponents = iter(self._exponents[k] for k in range(num_terms))
            return tuple((next(coeffs), k, next(exponents))
                         for k in range(self._num_regs)
                         for _ in range(self._num_terms_per_reg[k]))
        return None

    @property
    def scaling(self):
        if 'SCALED' not in PhaseFunc.FuncType(self._phase_func_type).name:
            return None
        return self._parameters[0]

    @property
    def divergence_override(self):
        if 'INVERSE' not in PhaseFunc.FuncType(self._phase_func_type).name:
            return None
        div_ind = 0
        if 'SCALED' in PhaseFunc.FuncType(self._phase_func_type).name:
            div_ind += 1
        return self._parameters[div_ind]

    @property
    def shifts(self):
        if 'SHIFTED' not in PhaseFunc.FuncType(self._phase_func_type).name:
            return None
        cdef int start_ind = 0
        cdef int k
        # The shifts are all parameters after possible scaling and
        # divergence override parameters.
        if 'SCALED' in PhaseFunc.FuncType(self._phase_func_type).name:
            start_ind += 1
        if 'INVERSE' in PhaseFunc.FuncType(self._phase_func_type).name:
            start_ind += 1
        cdef qreal[:] shift_arr = np.ndarray(self._num_parameters - start_ind,
                                             dtype=pyquest.core.np_qreal)
        for k in range(self._num_parameters - start_ind):
            shift_arr[k] = self._parameters[k + start_ind]
        return shift_arr.base

    @property
    def func_type(self):
        return self.FuncType(self._phase_func_type)

    @property
    def bit_encoding(self):
        return self.BitEncoding(self._bit_encoding)

    @property
    def inverse(self):
        constructor_args = {}
        constructor_args['func_type'] = self.func_type
        if self.terms:
            constructor_args['terms'] = tuple(
                (-term[0], term[1], term[2]) for term in self.terms)
        else:
            if not self.scaling:
                # Every FuncType except 'EXPONENTIAL_POLYNOMIAL' has a
                # 'SCALED' version which we must use to invert the sign.
                constructor_args['func_type'] = f"SCALED_{self.func_type.name}"
                constructor_args['scaling'] = -1
            else:
                constructor_args['scaling'] = -self.scaling
        if self.divergence_override:
            constructor_args['divergence_override'] = -self.divergence_override
        if self.overrides:
            constructor_args['overrides'] = {
                k: -v for k, v in self.overrides.items()}
        constructor_args['targets'] = self.targets
        constructor_args['bit_encoding'] = self.bit_encoding
        constructor_args['shifts'] = self.shifts
        return PhaseFunc(**constructor_args)

    cdef int apply_to(self, Qureg c_register) except -1:
        pass
        if self._is_poly:
            if self._num_regs == 0:
                quest.applyPhaseFuncOverrides(
                    c_register, self._qubits_in_regs, self._num_qubits_per_reg[0],
                    self._bit_encoding, self._coeffs, self._exponents,
                    self._num_terms, self._override_inds,
                    self._override_phases, self._num_overrides)
            else:
                quest.applyMultiVarPhaseFuncOverrides(
                    c_register, self._qubits_in_regs, self._num_qubits_per_reg,
                    self._num_regs, self._bit_encoding, self._coeffs,
                    self._exponents, self._num_terms_per_reg,
                    self._override_inds, self._override_phases,
                    self._num_overrides)
        else:
            quest.applyParamNamedPhaseFuncOverrides(
                c_register, self._qubits_in_regs, self._num_qubits_per_reg,
                self._num_regs, self._bit_encoding, self._phase_func_type,
                self._parameters, self._num_parameters, self._override_inds,
                self._override_phases, self._num_overrides)


cdef class QFT(MultiQubitOperator):

    def __cinit__(self, targets=None, target=None):
        self.TYPE = OP_TYPES.OP_QFT

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.applyQFT(c_register, self._targets, self._num_targets)


cdef class FullQFT(GlobalOperator):

    def __cinit__(self):
        self.TYPE = OP_TYPES.OP_FULL_QFT

    cdef int apply_to(self, Qureg c_register) except -1:
        quest.applyFullQFT(c_register)
