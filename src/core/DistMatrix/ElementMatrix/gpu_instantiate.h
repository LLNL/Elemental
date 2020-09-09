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
    INST_COPY_CTOR(T, U, V, Device::CPU, Device::GPU);  \
    INST_COPY_CTOR(T, U, V, Device::GPU, Device::CPU);  \
    INST_COPY_CTOR(T, U, V, Device::GPU, Device::GPU);  \
                                                        \
    INST_ASSIGN_OP(T, U, V, Device::CPU, Device::GPU);  \
    INST_ASSIGN_OP(T, U, V, Device::GPU, Device::CPU)

#define INST_DISTMATRIX_CLASS(T)                                        \
    template class DistMatrix<T, COLDIST, ROWDIST, ELEMENT, Device::GPU>; \
                                                                        \
    INST_COPY_CTOR(T, COLDIST, ROWDIST, Device::CPU, Device::GPU);      \
    INST_COPY_CTOR(T, COLDIST, ROWDIST, Device::GPU, Device::CPU);      \
                                                                        \
    INST_ASSIGN_OP(T, COLDIST, ROWDIST, Device::CPU, Device::GPU);      \
    INST_ASSIGN_OP(T, COLDIST, ROWDIST, Device::GPU, Device::CPU)
