// This will do the entire instantiation of GPU matrices for all known
// types, then undef everything.
//
// The following must be #define-d before #include-ing this file:
//   - COLDIST
//   - ROWDIST
//   - PROTO(T)
//
// The definitions of these provided macros will be intact at
// exit. The macros defined in this file will be all be #undef-d
// before exit.

#define INST_COPY_CTOR(T,U,V,SRCDEV,TGTDEV)                             \
  template DistMatrix<T, COLDIST, ROWDIST, ELEMENT, TGTDEV>::DistMatrix( \
    DistMatrix<T, U, V, ELEMENT, SRCDEV> const&)
#define INST_ASSIGN_OP(T,U,V,SRCDEV,TGTDEV)                     \
  template DistMatrix<T,COLDIST,ROWDIST,ELEMENT,TGTDEV>&        \
  DistMatrix<T, COLDIST, ROWDIST, ELEMENT, TGTDEV>::operator=   \
  (DistMatrix<T, U, V, ELEMENT, SRCDEV> const&)

#define INST_COPY_AND_ASSIGN(T, U, V)                   \
  INST_COPY_CTOR(T, U, V, Device::CPU, Device::GPU);    \
  INST_COPY_CTOR(T, U, V, Device::GPU, Device::CPU);    \
                                                        \
  INST_ASSIGN_OP(T, U, V, Device::CPU, Device::GPU);    \
  INST_ASSIGN_OP(T, U, V, Device::GPU, Device::CPU)

#define INST_DISTMATRIX_CLASS(T)                                        \
  template class DistMatrix<T, COLDIST, ROWDIST, ELEMENT, Device::GPU>

#define FULL_GPU_PROTO(T)                       \
  INST_DISTMATRIX_CLASS(T);                     \
  INST_COPY_AND_ASSIGN(T, CIRC, CIRC);          \
  INST_COPY_AND_ASSIGN(T, MC,   MR  );          \
  INST_COPY_AND_ASSIGN(T, MC,   STAR);          \
  INST_COPY_AND_ASSIGN(T, MD,   STAR);          \
  INST_COPY_AND_ASSIGN(T, MR,   MC  );          \
  INST_COPY_AND_ASSIGN(T, MR,   STAR);          \
  INST_COPY_AND_ASSIGN(T, STAR, MC  );          \
  INST_COPY_AND_ASSIGN(T, STAR, MD  );          \
  INST_COPY_AND_ASSIGN(T, STAR, MR  );          \
  INST_COPY_AND_ASSIGN(T, STAR, STAR);          \
  INST_COPY_AND_ASSIGN(T, STAR, VC  );          \
  INST_COPY_AND_ASSIGN(T, STAR, VR  );          \
  INST_COPY_AND_ASSIGN(T, VC,   STAR);          \
  INST_COPY_AND_ASSIGN(T, VR,   STAR)

#ifdef HYDROGEN_GPU_USE_FP16
PROTO(gpu_half_type)
FULL_GPU_PROTO(gpu_half_type);
#endif // HYDROGEN_GPU_USE_FP16

FULL_GPU_PROTO(float);
FULL_GPU_PROTO(double);
FULL_GPU_PROTO(El::Complex<float>);
FULL_GPU_PROTO(El::Complex<double>);

#undef FULL_GPU_PROTO
#undef INST_DISTMATRIX_CLASS
#undef INST_COPY_AND_ASSIGN
#undef INST_ASSIGN_OP
#undef INST_COPY_CTOR
