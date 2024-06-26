// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.add_eint' op should have the width of encrypted inputs equal
func.func @bad_inputs_width(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<3>) -> !FHE.eint<2> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<3>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// -----

// CHECK-LABEL: error: 'FHE.add_eint' op should have the signedness of encrypted inputs equal
func.func @bad_inputs_signedness(%arg0: !FHE.eint<2>, %arg1: !FHE.esint<2>) -> !FHE.eint<2> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.esint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// -----

// CHECK-LABEL: error: 'FHE.add_eint' op should have the width of encrypted inputs and result equal
func.func @bad_result_width(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<3> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

// -----

// CHECK-LABEL: error: 'FHE.add_eint' op should have the signedness of encrypted inputs and result equal
func.func @bad_result_signedness(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.esint<2> {
  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
}
