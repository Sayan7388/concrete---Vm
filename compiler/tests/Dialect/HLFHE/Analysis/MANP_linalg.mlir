// RUN: zamacompiler --passes MANP --action=dump-hlfhe --split-input-file %s 2>&1 | FileCheck %s

func @single_cst_add_eint_int(%t: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst = std.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint_int"(%t, %cst) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_dyn_add_eint_int(%e: tensor<8x!HLFHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint_int"(%e, %i) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_add_eint(%e0: tensor<8x!HLFHE.eint<2>>, %e1: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 2 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint"(%e0, %e1) : (tensor<8x!HLFHE.eint<2>>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_cst_sub_int_eint(%e: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst = std.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // CHECK: %[[ret:.*]] = "HLFHELinalg.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8xi3>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.sub_int_eint"(%cst, %e) : (tensor<8xi3>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_dyn_sub_int_eint(%e: tensor<8x!HLFHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.sub_int_eint"(%[[op0:.*]], %[[op1:.*]]) {MANP = 9 : ui{{[0-9]+}}} : (tensor<8xi3>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.sub_int_eint"(%i, %e) : (tensor<8xi3>, tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_cst_mul_eint_int(%e: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst = std.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>

  // %0 = "HLFHELinalg.mul_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 3 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.mul_eint_int"(%e, %cst) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @single_dyn_mul_eint_int(%e: tensor<8x!HLFHE.eint<2>>, %i: tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
{
  // CHECK: %[[ret:.*]] = "HLFHELinalg.mul_eint_int"([[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.mul_eint_int"(%e, %i) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>

  return %0 : tensor<8x!HLFHE.eint<2>>
}

// -----

func @chain_add_eint_int(%e: tensor<8x!HLFHE.eint<2>>) -> tensor<8x!HLFHE.eint<2>>
{
  %cst0 = std.constant dense<[0, 1, 2, 3, 3, 2, 1, 0]> : tensor<8xi3>
  %cst1 = std.constant dense<[0, 7, 2, 5, 6, 2, 1, 7]> : tensor<8xi3>
  %cst2 = std.constant dense<[0, 1, 2, 0, 1, 2, 0, 1]> : tensor<8xi3>
  %cst3 = std.constant dense<[0, 1, 1, 0, 0, 1, 0, 1]> : tensor<8xi3>
  // CHECK: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 4 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %0 = "HLFHELinalg.add_eint_int"(%e, %cst0) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %1 = "HLFHELinalg.add_eint_int"(%0, %cst1) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %2 = "HLFHELinalg.add_eint_int"(%1, %cst2) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  // CHECK-NEXT: %[[ret:.*]] = "HLFHELinalg.add_eint_int"(%[[op0:.*]], %[[op1:.*]]) {MANP = 8 : ui{{[0-9]+}}} : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  %3 = "HLFHELinalg.add_eint_int"(%2, %cst3) : (tensor<8x!HLFHE.eint<2>>, tensor<8xi3>) -> tensor<8x!HLFHE.eint<2>>
  return %3 : tensor<8x!HLFHE.eint<2>>
}
