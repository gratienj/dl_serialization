щИ
╔Ў
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718с┬
x
hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namehidden1/kernel
q
"hidden1/kernel/Read/ReadVariableOpReadVariableOphidden1/kernel*
_output_shapes

:*
dtype0
p
hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden1/bias
i
 hidden1/bias/Read/ReadVariableOpReadVariableOphidden1/bias*
_output_shapes
:*
dtype0
p
pswish1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namepswish1/beta
i
 pswish1/beta/Read/ReadVariableOpReadVariableOppswish1/beta*
_output_shapes
:*
dtype0
x
hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namehidden2/kernel
q
"hidden2/kernel/Read/ReadVariableOpReadVariableOphidden2/kernel*
_output_shapes

:*
dtype0
p
hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden2/bias
i
 hidden2/bias/Read/ReadVariableOpReadVariableOphidden2/bias*
_output_shapes
:*
dtype0
p
pswish2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namepswish2/beta
i
 pswish2/beta/Read/ReadVariableOpReadVariableOppswish2/beta*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
є
Adam/hidden1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/hidden1/kernel/m

)Adam/hidden1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/hidden1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/hidden1/bias/m
w
'Adam/hidden1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden1/bias/m*
_output_shapes
:*
dtype0
~
Adam/pswish1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/pswish1/beta/m
w
'Adam/pswish1/beta/m/Read/ReadVariableOpReadVariableOpAdam/pswish1/beta/m*
_output_shapes
:*
dtype0
є
Adam/hidden2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/hidden2/kernel/m

)Adam/hidden2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/hidden2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/hidden2/bias/m
w
'Adam/hidden2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden2/bias/m*
_output_shapes
:*
dtype0
~
Adam/pswish2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/pswish2/beta/m
w
'Adam/pswish2/beta/m/Read/ReadVariableOpReadVariableOpAdam/pswish2/beta/m*
_output_shapes
:*
dtype0
ѓ
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
є
Adam/hidden1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/hidden1/kernel/v

)Adam/hidden1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/hidden1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/hidden1/bias/v
w
'Adam/hidden1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden1/bias/v*
_output_shapes
:*
dtype0
~
Adam/pswish1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/pswish1/beta/v
w
'Adam/pswish1/beta/v/Read/ReadVariableOpReadVariableOpAdam/pswish1/beta/v*
_output_shapes
:*
dtype0
є
Adam/hidden2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/hidden2/kernel/v

)Adam/hidden2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/hidden2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/hidden2/bias/v
w
'Adam/hidden2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden2/bias/v*
_output_shapes
:*
dtype0
~
Adam/pswish2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/pswish2/beta/v
w
'Adam/pswish2/beta/v/Read/ReadVariableOpReadVariableOpAdam/pswish2/beta/v*
_output_shapes
:*
dtype0
ѓ
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"D░┐L  Dе ?»Ш >
`
Const_1Const*
_output_shapes
:*
dtype0*%
valueB"З#[Lй!CЕйЊ>ЕйЊ>

NoOpNoOp
Ш4
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*»4
valueЦ4Bб4 BЏ4
█
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
\
beta
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
\
$beta
%	variables
&regularization_losses
'trainable_variables
(	keras_api
R
)	variables
*regularization_losses
+trainable_variables
,	keras_api
h

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
л
3iter

4beta_1

5beta_2
	6decay
7learning_ratemkmlmmmnmo$mp-mq.mrvsvtvuvvvw$vx-vy.vz
8
0
1
2
3
4
$5
-6
.7
 
8
0
1
2
3
4
$5
-6
.7
Г
8metrics

9layers
:layer_metrics
;layer_regularization_losses

	variables
regularization_losses
trainable_variables
<non_trainable_variables
 
 
 
 
Г
=metrics
>layer_metrics

?layers
@layer_regularization_losses
	variables
regularization_losses
trainable_variables
Anon_trainable_variables
ZX
VARIABLE_VALUEhidden1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhidden1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
Bmetrics
Clayer_metrics

Dlayers
Elayer_regularization_losses
	variables
regularization_losses
trainable_variables
Fnon_trainable_variables
VT
VARIABLE_VALUEpswish1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
Г
Gmetrics
Hlayer_metrics

Ilayers
Jlayer_regularization_losses
	variables
regularization_losses
trainable_variables
Knon_trainable_variables
ZX
VARIABLE_VALUEhidden2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhidden2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
Lmetrics
Mlayer_metrics

Nlayers
Olayer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
Pnon_trainable_variables
VT
VARIABLE_VALUEpswish2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

$0
 

$0
Г
Qmetrics
Rlayer_metrics

Slayers
Tlayer_regularization_losses
%	variables
&regularization_losses
'trainable_variables
Unon_trainable_variables
 
 
 
Г
Vmetrics
Wlayer_metrics

Xlayers
Ylayer_regularization_losses
)	variables
*regularization_losses
+trainable_variables
Znon_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
Г
[metrics
\layer_metrics

]layers
^layer_regularization_losses
/	variables
0regularization_losses
1trainable_variables
_non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	btotal
	ccount
d	variables
e	keras_api
D
	ftotal
	gcount
h
_fn_kwargs
i	variables
j	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

b0
c1

d	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1

i	variables
}{
VARIABLE_VALUEAdam/hidden1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/hidden1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/pswish1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/hidden2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/hidden2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/pswish2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/hidden1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/hidden1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/pswish1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/hidden2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/hidden2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/pswish2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_inputPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
┼
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputConstConst_1hidden1/kernelhidden1/biaspswish1/betahidden2/kernelhidden2/biaspswish2/betadense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference_signature_wrapper_3728
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
■
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"hidden1/kernel/Read/ReadVariableOp hidden1/bias/Read/ReadVariableOp pswish1/beta/Read/ReadVariableOp"hidden2/kernel/Read/ReadVariableOp hidden2/bias/Read/ReadVariableOp pswish2/beta/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/hidden1/kernel/m/Read/ReadVariableOp'Adam/hidden1/bias/m/Read/ReadVariableOp'Adam/pswish1/beta/m/Read/ReadVariableOp)Adam/hidden2/kernel/m/Read/ReadVariableOp'Adam/hidden2/bias/m/Read/ReadVariableOp'Adam/pswish2/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/hidden1/kernel/v/Read/ReadVariableOp'Adam/hidden1/bias/v/Read/ReadVariableOp'Adam/pswish1/beta/v/Read/ReadVariableOp)Adam/hidden2/kernel/v/Read/ReadVariableOp'Adam/hidden2/bias/v/Read/ReadVariableOp'Adam/pswish2/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst_2*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *&
f!R
__inference__traced_save_4097
с
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden1/kernelhidden1/biaspswish1/betahidden2/kernelhidden2/biaspswish2/betadense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/hidden1/kernel/mAdam/hidden1/bias/mAdam/pswish1/beta/mAdam/hidden2/kernel/mAdam/hidden2/bias/mAdam/pswish2/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/hidden1/kernel/vAdam/hidden1/bias/vAdam/pswish1/beta/vAdam/hidden2/kernel/vAdam/hidden2/bias/vAdam/pswish2/beta/vAdam/dense/kernel/vAdam/dense/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *)
f$R"
 __inference__traced_restore_4206ъИ
џ

Л
(__inference_carnotnet_layer_call_fn_3829

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_carnotnet_layer_call_and_return_conditional_losses_34412
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
г"
ъ
C__inference_carnotnet_layer_call_and_return_conditional_losses_3585

inputs
scale_layer_3557
scale_layer_3559
hidden1_3562:
hidden1_3564:
pswish1_3567:
hidden2_3570:
hidden2_3572:
pswish2_3575:

dense_3579:

dense_3581:
identityѕбdense/StatefulPartitionedCallбhidden1/StatefulPartitionedCallбhidden2/StatefulPartitionedCallбpswish1/StatefulPartitionedCallбpswish2/StatefulPartitionedCallё
scale_layer/PartitionedCallPartitionedCallinputsscale_layer_3557scale_layer_3559*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_scale_layer_layer_call_and_return_conditional_losses_33512
scale_layer/PartitionedCallф
hidden1/StatefulPartitionedCallStatefulPartitionedCall$scale_layer/PartitionedCall:output:0hidden1_3562hidden1_3564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_33672!
hidden1/StatefulPartitionedCallъ
pswish1/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0pswish1_3567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish1_layer_call_and_return_conditional_losses_33822!
pswish1/StatefulPartitionedCall«
hidden2/StatefulPartitionedCallStatefulPartitionedCall(pswish1/StatefulPartitionedCall:output:0hidden2_3570hidden2_3572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_33962!
hidden2/StatefulPartitionedCallъ
pswish2/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0pswish2_3575*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish2_layer_call_and_return_conditional_losses_34112!
pswish2/StatefulPartitionedCallД
concatenate/PartitionedCallPartitionedCall$scale_layer/PartitionedCall:output:0(pswish2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_34222
concatenate/PartitionedCallа
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_3579
dense_3581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34342
dense/StatefulPartitionedCallб
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall ^pswish1/StatefulPartitionedCall ^pswish2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2B
pswish1/StatefulPartitionedCallpswish1/StatefulPartitionedCall2B
pswish2/StatefulPartitionedCallpswish2/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
ё
ў
A__inference_pswish1_layer_call_and_return_conditional_losses_3382

inputs%
readvariableop_resource:
identityѕбReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpc
mulMulReadVariableOp:value:0inputs*
T0*'
_output_shapes
:         2
mulX
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:         2	
Sigmoid\
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:         2
mul_1n
IdentityIdentity	mul_1:z:0^ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
С
q
E__inference_concatenate_layer_call_and_return_conditional_losses_3948
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЂ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
Ќ

л
(__inference_carnotnet_layer_call_fn_3464	
input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_carnotnet_layer_call_and_return_conditional_losses_34412
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput: 

_output_shapes
:: 

_output_shapes
:
Е+
▓
C__inference_carnotnet_layer_call_and_return_conditional_losses_3766

inputs
scale_layer_sub_y
scale_layer_truediv_y8
&hidden1_matmul_readvariableop_resource:5
'hidden1_biasadd_readvariableop_resource:-
pswish1_readvariableop_resource:8
&hidden2_matmul_readvariableop_resource:5
'hidden2_biasadd_readvariableop_resource:-
pswish2_readvariableop_resource:6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбhidden1/BiasAdd/ReadVariableOpбhidden1/MatMul/ReadVariableOpбhidden2/BiasAdd/ReadVariableOpбhidden2/MatMul/ReadVariableOpбpswish1/ReadVariableOpбpswish2/ReadVariableOpv
scale_layer/subSubinputsscale_layer_sub_y*
T0*'
_output_shapes
:         2
scale_layer/subЊ
scale_layer/truedivRealDivscale_layer/sub:z:0scale_layer_truediv_y*
T0*'
_output_shapes
:         2
scale_layer/truedivЦ
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
hidden1/MatMul/ReadVariableOpю
hidden1/MatMulMatMulscale_layer/truediv:z:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
hidden1/MatMulц
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden1/BiasAdd/ReadVariableOpА
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
hidden1/BiasAddї
pswish1/ReadVariableOpReadVariableOppswish1_readvariableop_resource*
_output_shapes
:*
dtype02
pswish1/ReadVariableOpЇ
pswish1/mulMulpswish1/ReadVariableOp:value:0hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
pswish1/mulp
pswish1/SigmoidSigmoidpswish1/mul:z:0*
T0*'
_output_shapes
:         2
pswish1/Sigmoidє
pswish1/mul_1Mulhidden1/BiasAdd:output:0pswish1/Sigmoid:y:0*
T0*'
_output_shapes
:         2
pswish1/mul_1Ц
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
hidden2/MatMul/ReadVariableOpќ
hidden2/MatMulMatMulpswish1/mul_1:z:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
hidden2/MatMulц
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden2/BiasAdd/ReadVariableOpА
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
hidden2/BiasAddї
pswish2/ReadVariableOpReadVariableOppswish2_readvariableop_resource*
_output_shapes
:*
dtype02
pswish2/ReadVariableOpЇ
pswish2/mulMulpswish2/ReadVariableOp:value:0hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
pswish2/mulp
pswish2/SigmoidSigmoidpswish2/mul:z:0*
T0*'
_output_shapes
:         2
pswish2/Sigmoidє
pswish2/mul_1Mulhidden2/BiasAdd:output:0pswish2/Sigmoid:y:0*
T0*'
_output_shapes
:         2
pswish2/mul_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisй
concatenate/concatConcatV2scale_layer/truediv:z:0pswish2/mul_1:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatenate/concatЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpџ
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdd█
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^hidden1/BiasAdd/ReadVariableOp^hidden1/MatMul/ReadVariableOp^hidden2/BiasAdd/ReadVariableOp^hidden2/MatMul/ReadVariableOp^pswish1/ReadVariableOp^pswish2/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
hidden1/BiasAdd/ReadVariableOphidden1/BiasAdd/ReadVariableOp2>
hidden1/MatMul/ReadVariableOphidden1/MatMul/ReadVariableOp2@
hidden2/BiasAdd/ReadVariableOphidden2/BiasAdd/ReadVariableOp2>
hidden2/MatMul/ReadVariableOphidden2/MatMul/ReadVariableOp20
pswish1/ReadVariableOppswish1/ReadVariableOp20
pswish2/ReadVariableOppswish2/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
ћ
Љ
$__inference_dense_layer_call_fn_3973

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34342
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ё
ў
A__inference_pswish2_layer_call_and_return_conditional_losses_3411

inputs%
readvariableop_resource:
identityѕбReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpc
mulMulReadVariableOp:value:0inputs*
T0*'
_output_shapes
:         2
mulX
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:         2	
Sigmoid\
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:         2
mul_1n
IdentityIdentity	mul_1:z:0^ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
{
E__inference_scale_layer_layer_call_and_return_conditional_losses_3351

inputs	
sub_y
	truediv_y
identityR
subSubinputssub_y*
T0*'
_output_shapes
:         2
subc
truedivRealDivsub:z:0	truediv_y*
T0*'
_output_shapes
:         2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
ў
Њ
&__inference_hidden1_layer_call_fn_3890

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_33672
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ў
Њ
&__inference_hidden2_layer_call_fn_3925

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_33962
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е"
Ю
C__inference_carnotnet_layer_call_and_return_conditional_losses_3664	
input
scale_layer_3636
scale_layer_3638
hidden1_3641:
hidden1_3643:
pswish1_3646:
hidden2_3649:
hidden2_3651:
pswish2_3654:

dense_3658:

dense_3660:
identityѕбdense/StatefulPartitionedCallбhidden1/StatefulPartitionedCallбhidden2/StatefulPartitionedCallбpswish1/StatefulPartitionedCallбpswish2/StatefulPartitionedCallЃ
scale_layer/PartitionedCallPartitionedCallinputscale_layer_3636scale_layer_3638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_scale_layer_layer_call_and_return_conditional_losses_33512
scale_layer/PartitionedCallф
hidden1/StatefulPartitionedCallStatefulPartitionedCall$scale_layer/PartitionedCall:output:0hidden1_3641hidden1_3643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_33672!
hidden1/StatefulPartitionedCallъ
pswish1/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0pswish1_3646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish1_layer_call_and_return_conditional_losses_33822!
pswish1/StatefulPartitionedCall«
hidden2/StatefulPartitionedCallStatefulPartitionedCall(pswish1/StatefulPartitionedCall:output:0hidden2_3649hidden2_3651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_33962!
hidden2/StatefulPartitionedCallъ
pswish2/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0pswish2_3654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish2_layer_call_and_return_conditional_losses_34112!
pswish2/StatefulPartitionedCallД
concatenate/PartitionedCallPartitionedCall$scale_layer/PartitionedCall:output:0(pswish2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_34222
concatenate/PartitionedCallа
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_3658
dense_3660*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34342
dense/StatefulPartitionedCallб
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall ^pswish1/StatefulPartitionedCall ^pswish2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2B
pswish1/StatefulPartitionedCallpswish1/StatefulPartitionedCall2B
pswish2/StatefulPartitionedCallpswish2/StatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput: 

_output_shapes
:: 

_output_shapes
:
═	
Ы
A__inference_hidden1_layer_call_and_return_conditional_losses_3367

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ч4
┴
__inference__wrapped_model_3336	
input
carnotnet_scale_layer_sub_y#
carnotnet_scale_layer_truediv_yB
0carnotnet_hidden1_matmul_readvariableop_resource:?
1carnotnet_hidden1_biasadd_readvariableop_resource:7
)carnotnet_pswish1_readvariableop_resource:B
0carnotnet_hidden2_matmul_readvariableop_resource:?
1carnotnet_hidden2_biasadd_readvariableop_resource:7
)carnotnet_pswish2_readvariableop_resource:@
.carnotnet_dense_matmul_readvariableop_resource:=
/carnotnet_dense_biasadd_readvariableop_resource:
identityѕб&carnotnet/dense/BiasAdd/ReadVariableOpб%carnotnet/dense/MatMul/ReadVariableOpб(carnotnet/hidden1/BiasAdd/ReadVariableOpб'carnotnet/hidden1/MatMul/ReadVariableOpб(carnotnet/hidden2/BiasAdd/ReadVariableOpб'carnotnet/hidden2/MatMul/ReadVariableOpб carnotnet/pswish1/ReadVariableOpб carnotnet/pswish2/ReadVariableOpЊ
carnotnet/scale_layer/subSubinputcarnotnet_scale_layer_sub_y*
T0*'
_output_shapes
:         2
carnotnet/scale_layer/sub╗
carnotnet/scale_layer/truedivRealDivcarnotnet/scale_layer/sub:z:0carnotnet_scale_layer_truediv_y*
T0*'
_output_shapes
:         2
carnotnet/scale_layer/truediv├
'carnotnet/hidden1/MatMul/ReadVariableOpReadVariableOp0carnotnet_hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'carnotnet/hidden1/MatMul/ReadVariableOp─
carnotnet/hidden1/MatMulMatMul!carnotnet/scale_layer/truediv:z:0/carnotnet/hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
carnotnet/hidden1/MatMul┬
(carnotnet/hidden1/BiasAdd/ReadVariableOpReadVariableOp1carnotnet_hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(carnotnet/hidden1/BiasAdd/ReadVariableOp╔
carnotnet/hidden1/BiasAddBiasAdd"carnotnet/hidden1/MatMul:product:00carnotnet/hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
carnotnet/hidden1/BiasAddф
 carnotnet/pswish1/ReadVariableOpReadVariableOp)carnotnet_pswish1_readvariableop_resource*
_output_shapes
:*
dtype02"
 carnotnet/pswish1/ReadVariableOpх
carnotnet/pswish1/mulMul(carnotnet/pswish1/ReadVariableOp:value:0"carnotnet/hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
carnotnet/pswish1/mulј
carnotnet/pswish1/SigmoidSigmoidcarnotnet/pswish1/mul:z:0*
T0*'
_output_shapes
:         2
carnotnet/pswish1/Sigmoid«
carnotnet/pswish1/mul_1Mul"carnotnet/hidden1/BiasAdd:output:0carnotnet/pswish1/Sigmoid:y:0*
T0*'
_output_shapes
:         2
carnotnet/pswish1/mul_1├
'carnotnet/hidden2/MatMul/ReadVariableOpReadVariableOp0carnotnet_hidden2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'carnotnet/hidden2/MatMul/ReadVariableOpЙ
carnotnet/hidden2/MatMulMatMulcarnotnet/pswish1/mul_1:z:0/carnotnet/hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
carnotnet/hidden2/MatMul┬
(carnotnet/hidden2/BiasAdd/ReadVariableOpReadVariableOp1carnotnet_hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(carnotnet/hidden2/BiasAdd/ReadVariableOp╔
carnotnet/hidden2/BiasAddBiasAdd"carnotnet/hidden2/MatMul:product:00carnotnet/hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
carnotnet/hidden2/BiasAddф
 carnotnet/pswish2/ReadVariableOpReadVariableOp)carnotnet_pswish2_readvariableop_resource*
_output_shapes
:*
dtype02"
 carnotnet/pswish2/ReadVariableOpх
carnotnet/pswish2/mulMul(carnotnet/pswish2/ReadVariableOp:value:0"carnotnet/hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
carnotnet/pswish2/mulј
carnotnet/pswish2/SigmoidSigmoidcarnotnet/pswish2/mul:z:0*
T0*'
_output_shapes
:         2
carnotnet/pswish2/Sigmoid«
carnotnet/pswish2/mul_1Mul"carnotnet/hidden2/BiasAdd:output:0carnotnet/pswish2/Sigmoid:y:0*
T0*'
_output_shapes
:         2
carnotnet/pswish2/mul_1ѕ
!carnotnet/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!carnotnet/concatenate/concat/axis№
carnotnet/concatenate/concatConcatV2!carnotnet/scale_layer/truediv:z:0carnotnet/pswish2/mul_1:z:0*carnotnet/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
carnotnet/concatenate/concatй
%carnotnet/dense/MatMul/ReadVariableOpReadVariableOp.carnotnet_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%carnotnet/dense/MatMul/ReadVariableOp┬
carnotnet/dense/MatMulMatMul%carnotnet/concatenate/concat:output:0-carnotnet/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
carnotnet/dense/MatMul╝
&carnotnet/dense/BiasAdd/ReadVariableOpReadVariableOp/carnotnet_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&carnotnet/dense/BiasAdd/ReadVariableOp┴
carnotnet/dense/BiasAddBiasAdd carnotnet/dense/MatMul:product:0.carnotnet/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
carnotnet/dense/BiasAddх
IdentityIdentity carnotnet/dense/BiasAdd:output:0'^carnotnet/dense/BiasAdd/ReadVariableOp&^carnotnet/dense/MatMul/ReadVariableOp)^carnotnet/hidden1/BiasAdd/ReadVariableOp(^carnotnet/hidden1/MatMul/ReadVariableOp)^carnotnet/hidden2/BiasAdd/ReadVariableOp(^carnotnet/hidden2/MatMul/ReadVariableOp!^carnotnet/pswish1/ReadVariableOp!^carnotnet/pswish2/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 2P
&carnotnet/dense/BiasAdd/ReadVariableOp&carnotnet/dense/BiasAdd/ReadVariableOp2N
%carnotnet/dense/MatMul/ReadVariableOp%carnotnet/dense/MatMul/ReadVariableOp2T
(carnotnet/hidden1/BiasAdd/ReadVariableOp(carnotnet/hidden1/BiasAdd/ReadVariableOp2R
'carnotnet/hidden1/MatMul/ReadVariableOp'carnotnet/hidden1/MatMul/ReadVariableOp2T
(carnotnet/hidden2/BiasAdd/ReadVariableOp(carnotnet/hidden2/BiasAdd/ReadVariableOp2R
'carnotnet/hidden2/MatMul/ReadVariableOp'carnotnet/hidden2/MatMul/ReadVariableOp2D
 carnotnet/pswish1/ReadVariableOp carnotnet/pswish1/ReadVariableOp2D
 carnotnet/pswish2/ReadVariableOp carnotnet/pswish2/ReadVariableOp:N J
'
_output_shapes
:         

_user_specified_nameinput: 

_output_shapes
:: 

_output_shapes
:
█
o
E__inference_concatenate_layer_call_and_return_conditional_losses_3422

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
Ќ

л
(__inference_carnotnet_layer_call_fn_3633	
input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_carnotnet_layer_call_and_return_conditional_losses_35852
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput: 

_output_shapes
:: 

_output_shapes
:
Е
{
E__inference_scale_layer_layer_call_and_return_conditional_losses_3862

inputs	
sub_y
	truediv_y
identityR
subSubinputssub_y*
T0*'
_output_shapes
:         2
subc
truedivRealDivsub:z:0	truediv_y*
T0*'
_output_shapes
:         2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
├
b
*__inference_scale_layer_layer_call_fn_3871

inputs
unknown
	unknown_0
identity▄
PartitionedCallPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_scale_layer_layer_call_and_return_conditional_losses_33512
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
ЄЇ
њ
 __inference__traced_restore_4206
file_prefix1
assignvariableop_hidden1_kernel:-
assignvariableop_1_hidden1_bias:-
assignvariableop_2_pswish1_beta:3
!assignvariableop_3_hidden2_kernel:-
assignvariableop_4_hidden2_bias:-
assignvariableop_5_pswish2_beta:1
assignvariableop_6_dense_kernel:+
assignvariableop_7_dense_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: ;
)assignvariableop_17_adam_hidden1_kernel_m:5
'assignvariableop_18_adam_hidden1_bias_m:5
'assignvariableop_19_adam_pswish1_beta_m:;
)assignvariableop_20_adam_hidden2_kernel_m:5
'assignvariableop_21_adam_hidden2_bias_m:5
'assignvariableop_22_adam_pswish2_beta_m:9
'assignvariableop_23_adam_dense_kernel_m:3
%assignvariableop_24_adam_dense_bias_m:;
)assignvariableop_25_adam_hidden1_kernel_v:5
'assignvariableop_26_adam_hidden1_bias_v:5
'assignvariableop_27_adam_pswish1_beta_v:;
)assignvariableop_28_adam_hidden2_kernel_v:5
'assignvariableop_29_adam_hidden2_bias_v:5
'assignvariableop_30_adam_pswish2_beta_v:9
'assignvariableop_31_adam_dense_kernel_v:3
%assignvariableop_32_adam_dense_bias_v:
identity_34ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9к
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*м
value╚B┼"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesм
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesп
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ъ
_output_shapesІ
ѕ::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityъ
AssignVariableOpAssignVariableOpassignvariableop_hidden1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ц
AssignVariableOp_1AssignVariableOpassignvariableop_1_hidden1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ц
AssignVariableOp_2AssignVariableOpassignvariableop_2_pswish1_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOp!assignvariableop_3_hidden2_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ц
AssignVariableOp_4AssignVariableOpassignvariableop_4_hidden2_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ц
AssignVariableOp_5AssignVariableOpassignvariableop_5_pswish2_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ц
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7б
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8А
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Б
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Д
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11д
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13А
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14А
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Б
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Б
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17▒
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_hidden1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18»
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_hidden1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19»
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_pswish1_beta_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20▒
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_hidden2_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21»
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_hidden2_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22»
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_pswish2_beta_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23»
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Г
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▒
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_hidden1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26»
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_hidden1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27»
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_pswish1_beta_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▒
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_hidden2_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29»
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_hidden2_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30»
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_pswish2_beta_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31»
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Г
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp┤
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33Д
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
в
v
&__inference_pswish2_layer_call_fn_3941

inputs
unknown:
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish2_layer_call_and_return_conditional_losses_34112
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
V
*__inference_concatenate_layer_call_fn_3954
inputs_0
inputs_1
identityМ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_34222
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
ё
ў
A__inference_pswish2_layer_call_and_return_conditional_losses_3934

inputs%
readvariableop_resource:
identityѕбReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpc
mulMulReadVariableOp:value:0inputs*
T0*'
_output_shapes
:         2
mulX
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:         2	
Sigmoid\
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:         2
mul_1n
IdentityIdentity	mul_1:z:0^ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
пF
ќ
__inference__traced_save_4097
file_prefix-
)savev2_hidden1_kernel_read_readvariableop+
'savev2_hidden1_bias_read_readvariableop+
'savev2_pswish1_beta_read_readvariableop-
)savev2_hidden2_kernel_read_readvariableop+
'savev2_hidden2_bias_read_readvariableop+
'savev2_pswish2_beta_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_hidden1_kernel_m_read_readvariableop2
.savev2_adam_hidden1_bias_m_read_readvariableop2
.savev2_adam_pswish1_beta_m_read_readvariableop4
0savev2_adam_hidden2_kernel_m_read_readvariableop2
.savev2_adam_hidden2_bias_m_read_readvariableop2
.savev2_adam_pswish2_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_hidden1_kernel_v_read_readvariableop2
.savev2_adam_hidden1_bias_v_read_readvariableop2
.savev2_adam_pswish1_beta_v_read_readvariableop4
0savev2_adam_hidden2_kernel_v_read_readvariableop2
.savev2_adam_hidden2_bias_v_read_readvariableop2
.savev2_adam_pswish2_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const_2

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename└
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*м
value╚B┼"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╠
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЃ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_hidden1_kernel_read_readvariableop'savev2_hidden1_bias_read_readvariableop'savev2_pswish1_beta_read_readvariableop)savev2_hidden2_kernel_read_readvariableop'savev2_hidden2_bias_read_readvariableop'savev2_pswish2_beta_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_hidden1_kernel_m_read_readvariableop.savev2_adam_hidden1_bias_m_read_readvariableop.savev2_adam_pswish1_beta_m_read_readvariableop0savev2_adam_hidden2_kernel_m_read_readvariableop.savev2_adam_hidden2_bias_m_read_readvariableop.savev2_adam_pswish2_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_hidden1_kernel_v_read_readvariableop.savev2_adam_hidden1_bias_v_read_readvariableop.savev2_adam_pswish1_beta_v_read_readvariableop0savev2_adam_hidden2_kernel_v_read_readvariableop.savev2_adam_hidden2_bias_v_read_readvariableop.savev2_adam_pswish2_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*▀
_input_shapes═
╩: ::::::::: : : : : : : : : ::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::"

_output_shapes
: 
в
v
&__inference_pswish1_layer_call_fn_3906

inputs
unknown:
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish1_layer_call_and_return_conditional_losses_33822
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
═	
Ы
A__inference_hidden1_layer_call_and_return_conditional_losses_3881

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е+
▓
C__inference_carnotnet_layer_call_and_return_conditional_losses_3804

inputs
scale_layer_sub_y
scale_layer_truediv_y8
&hidden1_matmul_readvariableop_resource:5
'hidden1_biasadd_readvariableop_resource:-
pswish1_readvariableop_resource:8
&hidden2_matmul_readvariableop_resource:5
'hidden2_biasadd_readvariableop_resource:-
pswish2_readvariableop_resource:6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбhidden1/BiasAdd/ReadVariableOpбhidden1/MatMul/ReadVariableOpбhidden2/BiasAdd/ReadVariableOpбhidden2/MatMul/ReadVariableOpбpswish1/ReadVariableOpбpswish2/ReadVariableOpv
scale_layer/subSubinputsscale_layer_sub_y*
T0*'
_output_shapes
:         2
scale_layer/subЊ
scale_layer/truedivRealDivscale_layer/sub:z:0scale_layer_truediv_y*
T0*'
_output_shapes
:         2
scale_layer/truedivЦ
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
hidden1/MatMul/ReadVariableOpю
hidden1/MatMulMatMulscale_layer/truediv:z:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
hidden1/MatMulц
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden1/BiasAdd/ReadVariableOpА
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
hidden1/BiasAddї
pswish1/ReadVariableOpReadVariableOppswish1_readvariableop_resource*
_output_shapes
:*
dtype02
pswish1/ReadVariableOpЇ
pswish1/mulMulpswish1/ReadVariableOp:value:0hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
pswish1/mulp
pswish1/SigmoidSigmoidpswish1/mul:z:0*
T0*'
_output_shapes
:         2
pswish1/Sigmoidє
pswish1/mul_1Mulhidden1/BiasAdd:output:0pswish1/Sigmoid:y:0*
T0*'
_output_shapes
:         2
pswish1/mul_1Ц
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
hidden2/MatMul/ReadVariableOpќ
hidden2/MatMulMatMulpswish1/mul_1:z:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
hidden2/MatMulц
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden2/BiasAdd/ReadVariableOpА
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
hidden2/BiasAddї
pswish2/ReadVariableOpReadVariableOppswish2_readvariableop_resource*
_output_shapes
:*
dtype02
pswish2/ReadVariableOpЇ
pswish2/mulMulpswish2/ReadVariableOp:value:0hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
pswish2/mulp
pswish2/SigmoidSigmoidpswish2/mul:z:0*
T0*'
_output_shapes
:         2
pswish2/Sigmoidє
pswish2/mul_1Mulhidden2/BiasAdd:output:0pswish2/Sigmoid:y:0*
T0*'
_output_shapes
:         2
pswish2/mul_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisй
concatenate/concatConcatV2scale_layer/truediv:z:0pswish2/mul_1:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatenate/concatЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpџ
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdd█
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^hidden1/BiasAdd/ReadVariableOp^hidden1/MatMul/ReadVariableOp^hidden2/BiasAdd/ReadVariableOp^hidden2/MatMul/ReadVariableOp^pswish1/ReadVariableOp^pswish2/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
hidden1/BiasAdd/ReadVariableOphidden1/BiasAdd/ReadVariableOp2>
hidden1/MatMul/ReadVariableOphidden1/MatMul/ReadVariableOp2@
hidden2/BiasAdd/ReadVariableOphidden2/BiasAdd/ReadVariableOp2>
hidden2/MatMul/ReadVariableOphidden2/MatMul/ReadVariableOp20
pswish1/ReadVariableOppswish1/ReadVariableOp20
pswish2/ReadVariableOppswish2/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
╦	
­
?__inference_dense_layer_call_and_return_conditional_losses_3434

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
џ

Л
(__inference_carnotnet_layer_call_fn_3854

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_carnotnet_layer_call_and_return_conditional_losses_35852
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
═	
Ы
A__inference_hidden2_layer_call_and_return_conditional_losses_3396

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦	
­
?__inference_dense_layer_call_and_return_conditional_losses_3964

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е"
Ю
C__inference_carnotnet_layer_call_and_return_conditional_losses_3695	
input
scale_layer_3667
scale_layer_3669
hidden1_3672:
hidden1_3674:
pswish1_3677:
hidden2_3680:
hidden2_3682:
pswish2_3685:

dense_3689:

dense_3691:
identityѕбdense/StatefulPartitionedCallбhidden1/StatefulPartitionedCallбhidden2/StatefulPartitionedCallбpswish1/StatefulPartitionedCallбpswish2/StatefulPartitionedCallЃ
scale_layer/PartitionedCallPartitionedCallinputscale_layer_3667scale_layer_3669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_scale_layer_layer_call_and_return_conditional_losses_33512
scale_layer/PartitionedCallф
hidden1/StatefulPartitionedCallStatefulPartitionedCall$scale_layer/PartitionedCall:output:0hidden1_3672hidden1_3674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_33672!
hidden1/StatefulPartitionedCallъ
pswish1/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0pswish1_3677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish1_layer_call_and_return_conditional_losses_33822!
pswish1/StatefulPartitionedCall«
hidden2/StatefulPartitionedCallStatefulPartitionedCall(pswish1/StatefulPartitionedCall:output:0hidden2_3680hidden2_3682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_33962!
hidden2/StatefulPartitionedCallъ
pswish2/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0pswish2_3685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish2_layer_call_and_return_conditional_losses_34112!
pswish2/StatefulPartitionedCallД
concatenate/PartitionedCallPartitionedCall$scale_layer/PartitionedCall:output:0(pswish2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_34222
concatenate/PartitionedCallа
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_3689
dense_3691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34342
dense/StatefulPartitionedCallб
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall ^pswish1/StatefulPartitionedCall ^pswish2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2B
pswish1/StatefulPartitionedCallpswish1/StatefulPartitionedCall2B
pswish2/StatefulPartitionedCallpswish2/StatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput: 

_output_shapes
:: 

_output_shapes
:
═	
Ы
A__inference_hidden2_layer_call_and_return_conditional_losses_3916

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
г"
ъ
C__inference_carnotnet_layer_call_and_return_conditional_losses_3441

inputs
scale_layer_3352
scale_layer_3354
hidden1_3368:
hidden1_3370:
pswish1_3383:
hidden2_3397:
hidden2_3399:
pswish2_3412:

dense_3435:

dense_3437:
identityѕбdense/StatefulPartitionedCallбhidden1/StatefulPartitionedCallбhidden2/StatefulPartitionedCallбpswish1/StatefulPartitionedCallбpswish2/StatefulPartitionedCallё
scale_layer/PartitionedCallPartitionedCallinputsscale_layer_3352scale_layer_3354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_scale_layer_layer_call_and_return_conditional_losses_33512
scale_layer/PartitionedCallф
hidden1/StatefulPartitionedCallStatefulPartitionedCall$scale_layer/PartitionedCall:output:0hidden1_3368hidden1_3370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_33672!
hidden1/StatefulPartitionedCallъ
pswish1/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0pswish1_3383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish1_layer_call_and_return_conditional_losses_33822!
pswish1/StatefulPartitionedCall«
hidden2/StatefulPartitionedCallStatefulPartitionedCall(pswish1/StatefulPartitionedCall:output:0hidden2_3397hidden2_3399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_33962!
hidden2/StatefulPartitionedCallъ
pswish2/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0pswish2_3412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_pswish2_layer_call_and_return_conditional_losses_34112!
pswish2/StatefulPartitionedCallД
concatenate/PartitionedCallPartitionedCall$scale_layer/PartitionedCall:output:0(pswish2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_34222
concatenate/PartitionedCallа
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0
dense_3435
dense_3437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34342
dense/StatefulPartitionedCallб
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall ^pswish1/StatefulPartitionedCall ^pswish2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2B
pswish1/StatefulPartitionedCallpswish1/StatefulPartitionedCall2B
pswish2/StatefulPartitionedCallpswish2/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs: 

_output_shapes
:: 

_output_shapes
:
ь	
╩
"__inference_signature_wrapper_3728	
input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *(
f#R!
__inference__wrapped_model_33362
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput: 

_output_shapes
:: 

_output_shapes
:
ё
ў
A__inference_pswish1_layer_call_and_return_conditional_losses_3899

inputs%
readvariableop_resource:
identityѕбReadVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpc
mulMulReadVariableOp:value:0inputs*
T0*'
_output_shapes
:         2
mulX
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:         2	
Sigmoid\
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:         2
mul_1n
IdentityIdentity	mul_1:z:0^ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         : 2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ц
serving_defaultљ
7
input.
serving_default_input:0         9
dense0
StatefulPartitionedCall:0         tensorflow/serving/predict:Вс
ћ;
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
regularization_losses
trainable_variables
	keras_api

signatures
*{&call_and_return_all_conditional_losses
|__call__
}_default_save_signature"▀7
_tf_keras_network├7{"name": "carnotnet", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "carnotnet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "ScaleLayer", "config": {"name": "scale_layer", "trainable": true, "dtype": "float32", "scale": [57446352.0, 152.1317901611328, 0.28855636715888977, 0.28855636715888977], "mean": [100500000.0, 536.5, 0.5000710487365723, 0.49992892146110535]}, "name": "scale_layer", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden1", "inbound_nodes": [[["scale_layer", 0, 0, {}]]]}, {"class_name": "PSwish", "config": {"name": "pswish1", "trainable": true, "dtype": "float32", "beta_initializer": {"class_name": "Ones", "config": {}}}, "name": "pswish1", "inbound_nodes": [[["hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden2", "inbound_nodes": [[["pswish1", 0, 0, {}]]]}, {"class_name": "PSwish", "config": {"name": "pswish2", "trainable": true, "dtype": "float32", "beta_initializer": {"class_name": "Ones", "config": {}}}, "name": "pswish2", "inbound_nodes": [[["hidden2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["scale_layer", 0, 0, {}], ["pswish2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense", 0, 0]]}, "shared_object_id": 16, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 4]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 4]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "carnotnet", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "ScaleLayer", "config": {"name": "scale_layer", "trainable": true, "dtype": "float32", "scale": [57446352.0, 152.1317901611328, 0.28855636715888977, 0.28855636715888977], "mean": [100500000.0, 536.5, 0.5000710487365723, 0.49992892146110535]}, "name": "scale_layer", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden1", "inbound_nodes": [[["scale_layer", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "PSwish", "config": {"name": "pswish1", "trainable": true, "dtype": "float32", "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}}, "name": "pswish1", "inbound_nodes": [[["hidden1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden2", "inbound_nodes": [[["pswish1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "PSwish", "config": {"name": "pswish2", "trainable": true, "dtype": "float32", "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 10}}, "name": "pswish2", "inbound_nodes": [[["hidden2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["scale_layer", 0, 0, {}], ["pswish2", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 15}], "input_layers": [["input", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": "mae", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 18}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
т"Р
_tf_keras_input_layer┬{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
Ў
	variables
regularization_losses
trainable_variables
	keras_api
*~&call_and_return_all_conditional_losses
__call__"і
_tf_keras_layer­{"name": "scale_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ScaleLayer", "config": {"name": "scale_layer", "trainable": true, "dtype": "float32", "scale": [57446352.0, 152.1317901611328, 0.28855636715888977, 0.28855636715888977], "mean": [100500000.0, 536.5, 0.5000710487365723, 0.49992892146110535]}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}
Щ

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+ђ&call_and_return_all_conditional_losses
Ђ__call__"М
_tf_keras_layer╣{"name": "hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["scale_layer", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
џ
beta
	variables
regularization_losses
trainable_variables
	keras_api
+ѓ&call_and_return_all_conditional_losses
Ѓ__call__" 
_tf_keras_layerт{"name": "pswish1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PSwish", "config": {"name": "pswish1", "trainable": true, "dtype": "float32", "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}}, "inbound_nodes": [[["hidden1", 0, 0, {}]]], "shared_object_id": 6, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
Э

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+ё&call_and_return_all_conditional_losses
Ё__call__"Л
_tf_keras_layerи{"name": "hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pswish1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ю
$beta
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+є&call_and_return_all_conditional_losses
Є__call__"Ђ
_tf_keras_layerу{"name": "pswish2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PSwish", "config": {"name": "pswish2", "trainable": true, "dtype": "float32", "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 10}}, "inbound_nodes": [[["hidden2", 0, 0, {}]]], "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
Е
)	variables
*regularization_losses
+trainable_variables
,	keras_api
+ѕ&call_and_return_all_conditional_losses
Ѕ__call__"ў
_tf_keras_layer■{"name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["scale_layer", 0, 0, {}], ["pswish2", 0, 0, {}]]], "shared_object_id": 12, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 4]}, {"class_name": "TensorShape", "items": [null, 16]}]}
■

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
+і&call_and_return_all_conditional_losses
І__call__"О
_tf_keras_layerй{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
с
3iter

4beta_1

5beta_2
	6decay
7learning_ratemkmlmmmnmo$mp-mq.mrvsvtvuvvvw$vx-vy.vz"
	optimizer
X
0
1
2
3
4
$5
-6
.7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
$5
-6
.7"
trackable_list_wrapper
╩
8metrics

9layers
:layer_metrics
;layer_regularization_losses

	variables
regularization_losses
trainable_variables
<non_trainable_variables
|__call__
}_default_save_signature
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
-
їserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
=metrics
>layer_metrics

?layers
@layer_regularization_losses
	variables
regularization_losses
trainable_variables
Anon_trainable_variables
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 :2hidden1/kernel
:2hidden1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Bmetrics
Clayer_metrics

Dlayers
Elayer_regularization_losses
	variables
regularization_losses
trainable_variables
Fnon_trainable_variables
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
:2pswish1/beta
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
░
Gmetrics
Hlayer_metrics

Ilayers
Jlayer_regularization_losses
	variables
regularization_losses
trainable_variables
Knon_trainable_variables
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 :2hidden2/kernel
:2hidden2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Lmetrics
Mlayer_metrics

Nlayers
Olayer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
Pnon_trainable_variables
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
:2pswish2/beta
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
$0"
trackable_list_wrapper
░
Qmetrics
Rlayer_metrics

Slayers
Tlayer_regularization_losses
%	variables
&regularization_losses
'trainable_variables
Unon_trainable_variables
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Vmetrics
Wlayer_metrics

Xlayers
Ylayer_regularization_losses
)	variables
*regularization_losses
+trainable_variables
Znon_trainable_variables
Ѕ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
░
[metrics
\layer_metrics

]layers
^layer_regularization_losses
/	variables
0regularization_losses
1trainable_variables
_non_trainable_variables
І__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
`0
a1"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
	btotal
	ccount
d	variables
e	keras_api"Ю
_tf_keras_metricѓ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 22}
І
	ftotal
	gcount
h
_fn_kwargs
i	variables
j	keras_api"─
_tf_keras_metricЕ{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 18}
:  (2total
:  (2count
.
b0
c1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
%:#2Adam/hidden1/kernel/m
:2Adam/hidden1/bias/m
:2Adam/pswish1/beta/m
%:#2Adam/hidden2/kernel/m
:2Adam/hidden2/bias/m
:2Adam/pswish2/beta/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#2Adam/hidden1/kernel/v
:2Adam/hidden1/bias/v
:2Adam/pswish1/beta/v
%:#2Adam/hidden2/kernel/v
:2Adam/hidden2/bias/v
:2Adam/pswish2/beta/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
┌2О
C__inference_carnotnet_layer_call_and_return_conditional_losses_3766
C__inference_carnotnet_layer_call_and_return_conditional_losses_3804
C__inference_carnotnet_layer_call_and_return_conditional_losses_3664
C__inference_carnotnet_layer_call_and_return_conditional_losses_3695└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
(__inference_carnotnet_layer_call_fn_3464
(__inference_carnotnet_layer_call_fn_3829
(__inference_carnotnet_layer_call_fn_3854
(__inference_carnotnet_layer_call_fn_3633└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
█2п
__inference__wrapped_model_3336┤
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *$б!
і
input         
№2В
E__inference_scale_layer_layer_call_and_return_conditional_losses_3862б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_scale_layer_layer_call_fn_3871б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_hidden1_layer_call_and_return_conditional_losses_3881б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_hidden1_layer_call_fn_3890б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_pswish1_layer_call_and_return_conditional_losses_3899б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_pswish1_layer_call_fn_3906б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_hidden2_layer_call_and_return_conditional_losses_3916б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_hidden2_layer_call_fn_3925б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_pswish2_layer_call_and_return_conditional_losses_3934б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_pswish2_layer_call_fn_3941б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_concatenate_layer_call_and_return_conditional_losses_3948б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_concatenate_layer_call_fn_3954б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ж2Т
?__inference_dense_layer_call_and_return_conditional_losses_3964б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╬2╦
$__inference_dense_layer_call_fn_3973б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
КB─
"__inference_signature_wrapper_3728input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
	J
Const
J	
Const_1љ
__inference__wrapped_model_3336mЇј$-..б+
$б!
і
input         
ф "-ф*
(
denseі
dense         ┤
C__inference_carnotnet_layer_call_and_return_conditional_losses_3664mЇј$-.6б3
,б)
і
input         
p 

 
ф "%б"
і
0         
џ ┤
C__inference_carnotnet_layer_call_and_return_conditional_losses_3695mЇј$-.6б3
,б)
і
input         
p

 
ф "%б"
і
0         
џ х
C__inference_carnotnet_layer_call_and_return_conditional_losses_3766nЇј$-.7б4
-б*
 і
inputs         
p 

 
ф "%б"
і
0         
џ х
C__inference_carnotnet_layer_call_and_return_conditional_losses_3804nЇј$-.7б4
-б*
 і
inputs         
p

 
ф "%б"
і
0         
џ ї
(__inference_carnotnet_layer_call_fn_3464`Їј$-.6б3
,б)
і
input         
p 

 
ф "і         ї
(__inference_carnotnet_layer_call_fn_3633`Їј$-.6б3
,б)
і
input         
p

 
ф "і         Ї
(__inference_carnotnet_layer_call_fn_3829aЇј$-.7б4
-б*
 і
inputs         
p 

 
ф "і         Ї
(__inference_carnotnet_layer_call_fn_3854aЇј$-.7б4
-б*
 і
inputs         
p

 
ф "і         ═
E__inference_concatenate_layer_call_and_return_conditional_losses_3948ЃZбW
PбM
KџH
"і
inputs/0         
"і
inputs/1         
ф "%б"
і
0         
џ ц
*__inference_concatenate_layer_call_fn_3954vZбW
PбM
KџH
"і
inputs/0         
"і
inputs/1         
ф "і         Ъ
?__inference_dense_layer_call_and_return_conditional_losses_3964\-./б,
%б"
 і
inputs         
ф "%б"
і
0         
џ w
$__inference_dense_layer_call_fn_3973O-./б,
%б"
 і
inputs         
ф "і         А
A__inference_hidden1_layer_call_and_return_conditional_losses_3881\/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
&__inference_hidden1_layer_call_fn_3890O/б,
%б"
 і
inputs         
ф "і         А
A__inference_hidden2_layer_call_and_return_conditional_losses_3916\/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ y
&__inference_hidden2_layer_call_fn_3925O/б,
%б"
 і
inputs         
ф "і         а
A__inference_pswish1_layer_call_and_return_conditional_losses_3899[/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ x
&__inference_pswish1_layer_call_fn_3906N/б,
%б"
 і
inputs         
ф "і         а
A__inference_pswish2_layer_call_and_return_conditional_losses_3934[$/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ x
&__inference_pswish2_layer_call_fn_3941N$/б,
%б"
 і
inputs         
ф "і         Д
E__inference_scale_layer_layer_call_and_return_conditional_losses_3862^Їј/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ 
*__inference_scale_layer_layer_call_fn_3871QЇј/б,
%б"
 і
inputs         
ф "і         ю
"__inference_signature_wrapper_3728vЇј$-.7б4
б 
-ф*
(
inputі
input         "-ф*
(
denseі
dense         