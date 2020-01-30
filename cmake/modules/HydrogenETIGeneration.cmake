# This function sets the _ARG# variable and recurs through the parameters.
function (h_gtpe_recur OUTVAR EXPRESSION_TEMPLATE THIS_EPARAM_ID THIS_EPARAM)
  foreach (_VAL IN LISTS ${THIS_EPARAM})
    set(_ARG${THIS_EPARAM_ID} "${_VAL}")
    if (ARGN)
      math(EXPR _NEXT_ID "${THIS_EPARAM_ID} + 1")
      h_gtpe_recur(${OUTVAR} "${EXPRESSION_TEMPLATE}" ${_NEXT_ID} ${ARGN})
    else ()
      string(CONFIGURE "${EXPRESSION_TEMPLATE}" _THIS_EXPRESSION @ONLY)
      list(APPEND ${OUTVAR} "${_THIS_EXPRESSION}")
    endif ()
  endforeach ()
  set(${OUTVAR} "${${OUTVAR}}" PARENT_SCOPE)
endfunction ()

# This function calls the recursive function above to generate each
# function instance.
#
# DO NOT ADD A SEMICOLON TO THE EXPRESSION_TEMPLATE PARAMETER!!!!!!!
function (h_generate_tensor_product_expression OUTVAR EXPRESSION_TEMPLATE)
  h_gtpe_recur(_ALL_EXPRESSIONS "${EXPRESSION_TEMPLATE}" 0 ${ARGN})
  set(${OUTVAR} ${_ALL_EXPRESSIONS} PARENT_SCOPE)
endfunction ()

# This function adds a semicolon to each function instance in the list
# and joins them into a string with each function instance on its own
# line.
function (h_func_list_to_string OUTVAR INLIST)
  list(JOIN ${INLIST} ";\n" _TMP)
  set(${OUTVAR} "${_TMP};" PARENT_SCOPE)
endfunction ()
